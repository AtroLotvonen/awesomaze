// q_learning.rs

use rayon::prelude::*;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
};
use tracing::error;

use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};

use crate::machine_learning::{
    nn::dimensions::Dimensions4D,
    reinforcement_learning::{
        environment::{RLAction, RLEnvironment, Replay},
        replay_buffer::ReplayBuffer,
        rl_error::RLResult,
        rl_model::{RLLoss, RLModel, RLTensor},
    },
};

pub trait DQNTargetModel {
    fn soft_update(&mut self, other: &Self, tau: f32);
}

pub struct DQNet<B>
where
    B: Clone,
{
    backend: B,
}

struct TrainBatch {
    pub prev_state_batch: Vec<f32>,
    pub action_batch: Vec<f32>,
    pub reward_batch: Vec<f32>,
    pub next_state_batch: Vec<f32>,
    pub terminal_batch: Vec<f32>,
}

pub struct LearningRateScheduler {
    lr: f32,
    step_size: f32,
    curr_step: usize,
    update_interval: usize,
    warmup: usize,
}

impl LearningRateScheduler {
    pub fn new(initial: f32, step_size: f32, update_interval: usize, warmup: usize) -> Self {
        Self {
            lr: initial,
            step_size,
            curr_step: 0,
            update_interval,
            warmup,
        }
    }
    pub fn step(&mut self) -> f32 {
        self.curr_step += 1;
        if self.curr_step < self.warmup && self.warmup != 0 {
            self.lr * (self.curr_step as f32 / self.warmup as f32)
        } else if self.curr_step % self.update_interval == 0 {
            self.lr *= self.step_size;
            self.lr
        } else {
            self.lr
        }
    }
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        Self {
            lr: 0.0001,
            step_size: 0.98,
            curr_step: 0,
            update_interval: 100000,
            warmup: 4000,
        }
    }
}

pub struct DQNetTrainingConfig {
    pub parallel_env: usize,
    pub max_steps: usize,
    pub patience: f32,
    pub batch_size: usize,
    pub epsilon: f32,
    pub tau: f32,
    pub replay_buffer_size: usize,
    pub gamma: f32,
    pub load_model: Option<PathBuf>,
    pub save_model: PathBuf,
}

impl Default for DQNetTrainingConfig {
    fn default() -> Self {
        Self {
            parallel_env: 64,
            max_steps: 1000000,
            patience: -2.0,
            batch_size: 64,
            epsilon: 0.1,
            tau: 0.005,
            replay_buffer_size: 100000,
            gamma: 0.99,
            load_model: None,
            save_model: "awesom".into(),
        }
    }
}

impl<B> DQNet<B>
where
    // E: RLEnvironment + Clone,
    // M: RLModel<B, T, O> + DQNTargetModel,
    // T: RLTensor<B>,
    B: Clone,
{
    pub fn new(backend: B) -> Self {
        Self {
            // environments,
            backend,
        }
    }

    pub fn train_loop<F, P, E, M, T, O>(
        &mut self,
        pre_step_hook: F,
        post_step_hook: P,
        training_config: DQNetTrainingConfig,
        mut optimizer: O,
        mut lr_scheduler: LearningRateScheduler,
    ) -> RLResult<()>
    where
        F: Fn(E, Vec<f32>) -> bool + Send + 'static,
        P: Fn(usize, f32, f32, Option<T>, T) -> Option<T>,
        E: RLEnvironment + Clone,
        M: RLModel<B, T, O> + DQNTargetModel,
        T: RLTensor<B>,
    {
        let mut replay_buffer: ReplayBuffer<E> =
            ReplayBuffer::new(training_config.replay_buffer_size);
        let mut environments = Vec::with_capacity(training_config.parallel_env);
        for _ in 0..training_config.parallel_env {
            environments.push(E::init());
        }
        let env_dims =
            Dimensions4D::from_d3(training_config.parallel_env, environments[0].dimensions());
        let train_dims =
            Dimensions4D::from_d3(training_config.batch_size, environments[0].dimensions());
        let mut model = M::init(
            training_config.load_model,
            self.backend.clone(),
            environments[0].dimensions(),
            E::Action::SIZE,
        )?;
        let mut target_model = model.clone();

        let (replay_tx, replay_rx) = mpsc::channel();
        let (train_tx, train_rx) = mpsc::channel();
        let (action_tx, action_rx) = mpsc::channel::<Vec<_>>(); // why does this need this
        let (pred_tx, pred_rx) = mpsc::channel();

        let parallel_count = training_config.parallel_env;

        let terminate = Arc::new(AtomicBool::new(false));
        let replay_terminate = Arc::clone(&terminate);

        // replay buffer thread
        let replay_thread_handle = thread::spawn(move || {
            let mut time_step = 0;
            while !replay_terminate.load(Ordering::Relaxed) {
                if time_step != 10 {
                    // read the new replays and add to the replay buffer
                    for _ in 0..parallel_count {
                        //info!("REPLAY THREAD: Waiting {i} replay.");
                        if let Ok(replay) = replay_rx.recv() {
                            //info!("REPLAY THREAD: Read {i} replay.");
                            replay_buffer.add(replay);
                        }
                    }
                }
                // sample replay values and preprocess them to vectors which can be made to tensors
                // easily
                let replays = replay_buffer.get(training_config.batch_size);
                let prev_state_batch: Vec<f32> = replays
                    .iter()
                    .flat_map(|&r| r.prev_state.as_ref())
                    .copied()
                    .collect();
                let action_batch: Vec<f32> = replays.iter().map(|&r| r.action.into()).collect();
                let reward_batch: Vec<f32> = replays.iter().map(|&r| r.reward).collect();
                let next_state_batch: Vec<f32> = replays
                    .iter()
                    .flat_map(|&r| r.next_state.as_ref())
                    .copied()
                    .collect();
                let terminal_batch: Vec<f32> = replays
                    .iter()
                    .map(|&r| if r.terminal { 0.0 } else { 1.0 }) // map to multiplier for train
                    .collect();
                let train_batch = TrainBatch {
                    prev_state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    terminal_batch,
                };
                //info!("REPLAY THREAD: Sent train batch.");
                if let Err(_err) = train_tx.send(train_batch) {
                    break;
                }
                time_step += 1;
            }
        });

        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(6)
        //     .build_global()
        //     .unwrap();

        let env_thread_handle = thread::spawn(move || {
            // initial states here, these are updated from the env step function
            let mut curr_states = environments.iter().map(|e| e.state()).collect::<Vec<_>>();
            let curr_states_batch = curr_states
                .iter()
                .flat_map(|state| state.as_ref())
                .copied()
                .collect::<Vec<_>>();

            if let Err(_error) = pred_tx.send(curr_states_batch) {
                error!("ENV THREAD: Failed sending current states.");
            }
            let mut episode_rewards = Vec::new();

            loop {
                let first_env = environments[0].clone();
                if pre_step_hook(first_env, episode_rewards) {
                    println!("stopping");
                    break;
                }
                //info!("ENV THREAD: Waiting actions.");
                if let Ok(actions) = action_rx.recv() {
                    // Environments actions
                    let curr_states_with_index = environments
                        .par_iter_mut()
                        .enumerate()
                        .zip(curr_states.into_par_iter())
                        .zip(actions.into_par_iter())
                        .map(|(((i, e), prev_state), action)| {
                            let policy_action =
                                Self::action_policy::<E>(action, training_config.epsilon);
                            let step = e.record_step(policy_action);
                            let next_state = match step.3 {
                                true => {
                                    let episode_reward = e.cumulative_reward();
                                    e.reset();
                                    (e.state(), Some(episode_reward))
                                }
                                false => {
                                    if e.cumulative_reward() < training_config.patience {
                                        let episode_reward = e.cumulative_reward();
                                        e.reset();
                                        (e.state(), Some(episode_reward))
                                    } else {
                                        (step.2.clone(), None)
                                    }
                                }
                            };
                            // Get the next state for prediction after reset
                            let replay = Replay {
                                prev_state,
                                action: step.0,
                                reward: step.1,
                                next_state: step.2,
                                terminal: step.3,
                            };
                            //info!("ENV THREAD: Sent replay for env number {i}.");
                            let _ = replay_tx.send(replay);
                            (i, next_state)
                        })
                        .collect::<Vec<_>>();
                    // Save the current states for next step
                    episode_rewards = curr_states_with_index
                        .iter()
                        .filter_map(|(_, state)| state.1)
                        .collect::<Vec<_>>();
                    curr_states = curr_states_with_index
                        .into_iter()
                        .sorted_by_key(|(i, _state)| *i)
                        .map(|(_, state)| state.0)
                        .collect::<Vec<_>>();
                    // Send the curr_states as batch for the policy prediction
                    let curr_state_batch = curr_states
                        .iter()
                        .flat_map(|state| state.as_ref())
                        .copied()
                        .collect::<Vec<_>>();
                    //info!("ENV THREAD: Sent env batch for prediction.");
                    let _ = pred_tx.send(curr_state_batch);
                } else {
                    error!("ENV THREAD: Failed receiving actions.");
                    break;
                }
            }
        });

        let mut metrics_saver = None;
        for step in 0..training_config.max_steps {
            // model thread
            // use the current policy to predict next actions
            //info!("MODEL THREAD: Waiting for current states in step {step}.");
            match pred_rx.recv() {
                Ok(curr_states) => {
                    //info!("MODEL THREAD: Predicting current step {step}.");
                    let curr_state_tensor =
                        RLTensor::from_values(curr_states, env_dims, &self.backend, false);
                    let actions = model
                        .predict(curr_state_tensor)
                        .arg_max(3)
                        .to_values()
                        .into_iter()
                        .map(|action| action.into())
                        .collect::<Vec<E::Action>>();
                    //info!("MODEL THREAD: Predicted current step {step}.");
                    if let Err(_err) = action_tx.send(actions) {
                        error!("MODEL THREAD: Sending actions for step failed {step}");
                        break;
                    }
                    //info!("MODEL THREAD: Sent current step {step} successfully.");
                }
                Err(_err) => {
                    error!("MODEL THREAD: Failed receiving actions for prediction {step}");
                    break;
                }
            }
            //info!("MODEL THREAD: Waiting for training batch in step {step}.");
            match train_rx.recv() {
                Ok(train_batch) => {
                    //info!("MODEL THREAD: Training batch in step {step}.");
                    let prev_state_tensor = T::from_values(
                        train_batch.prev_state_batch,
                        train_dims,
                        &self.backend,
                        false,
                    );
                    let action_tensor = T::from_values(
                        train_batch.action_batch,
                        Dimensions4D::new(training_config.batch_size, 1, 1, 1),
                        &self.backend,
                        false,
                    );
                    let reward_tensor = T::from_values(
                        train_batch.reward_batch,
                        Dimensions4D::new(training_config.batch_size, 1, 1, 1),
                        &self.backend,
                        false,
                    );
                    let next_state_tensor = T::from_values(
                        train_batch.next_state_batch,
                        train_dims,
                        &self.backend,
                        false,
                    );
                    let terminal_tensor = T::from_values(
                        train_batch.terminal_batch,
                        Dimensions4D::new(training_config.batch_size, 1, 1, 1),
                        &self.backend,
                        false,
                    );
                    let next_q = target_model.predict(next_state_tensor).max_dim(3).detach();
                    let prev_q = model.predict(prev_state_tensor).gather(3, action_tensor);
                    let expected = next_q
                        .mul(terminal_tensor)
                        .mul_scalar(training_config.gamma)
                        .add(reward_tensor);

                    // let pred_values = prev_q.to_values();
                    // let gt_values = expected.to_values();
                    // for (pred, gt) in pred_values.into_iter().zip(gt_values.into_iter()) {
                    //     eprintln!("🐞DEBUGPRINT[4]: tensor.rs:869: pred_values pred_values={:?}, gt_values={:?}",pred, gt);
                    // }
                    let loss = prev_q.loss(expected, RLLoss::Mse);
                    let lr = lr_scheduler.step();
                    model.optimize(loss.clone(), &mut optimizer, lr, step + 1);
                    target_model.soft_update(&model, training_config.tau);

                    // Save metrics from training
                    metrics_saver =
                        post_step_hook(step, lr, training_config.epsilon, metrics_saver, loss);
                }
                Err(_err) => {
                    error!("MODEL THREAD: Failed receiving training batch {step}");
                    break;
                }
            }
        }

        terminate.store(true, Ordering::Relaxed);

        let _ = model.save_model(training_config.save_model);

        let _ = env_thread_handle.join();
        let _ = replay_thread_handle.join();

        Ok(())
    }

    fn action_policy<E>(action: E::Action, exploration: f32) -> E::Action
    where
        E: RLEnvironment + Clone,
    {
        let action_space = E::Action::SIZE;
        let mut rng = rand::thread_rng();
        let explore = rng.gen_bool(exploration as f64);
        let action_u32: u32 = action.into();
        let action: E::Action = if explore {
            // get a random action not chosen by the policy
            (0..action_space)
                .filter(|&i| i != action_u32)
                .choose(&mut rng)
                .expect("Too few actions in the environment?")
                .into()
        } else {
            action
        };
        action
    }
}
