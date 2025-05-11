use std::cell::RefCell;
use std::io;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use awesomlearn::machine_learning::nn;
use awesomlearn::machine_learning::nn::backends::wgpu_backend::WgpuBackend;
use awesomlearn::machine_learning::nn::dimensions::Dimensions4D;
use awesomlearn::machine_learning::nn::optimizer::AdamOptimizer;
use awesomlearn::machine_learning::reinforcement_learning::q_learning::dqnet::{
    DQNet, DQNetTrainingConfig, LearningRateScheduler,
};
use awesomlearn::machine_learning::reinforcement_learning::rl_model::RLTensor;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamWConfig;
use burn::tensor::Tensor;
use ratatui::{
    layout::{Constraint, Layout},
    Frame,
};

use crate::ModelType;

use super::game_model::RLGameModel;
use super::model::awesom_model::AwesomModel;
use super::model::burn_model::{BurnTensor, Model};
use super::{engine::Engine, game_model::GameModel, info::Info, input::GameInput};

pub type ModelUpdate = Option<(GameModel, Vec<f32>)>;

pub struct TrainingEngine<const W: usize, const H: usize> {
    agent_engine_handle: Option<thread::JoinHandle<()>>,
    latest_model_receiver: Arc<Mutex<ModelUpdate>>,
    main_to_agent_sender: Sender<GameInput>,
    train_info_rx: Receiver<(f32, f32, f32)>,
    agent_visual_model: GameModel,
    agent_visual_info: Info,
    update_interval: usize,
}

impl<const W: usize, const H: usize> TrainingEngine<W, H> {
    pub fn new(
        config: DQNetTrainingConfig,
        schedule: LearningRateScheduler,
        model_type: ModelType,
    ) -> Self {
        let (main_to_agent_sender, main_to_agent_receiver) = mpsc::channel();
        let latest_model_receiver = Arc::new(Mutex::new(None));
        let latest_model_sender = Arc::clone(&latest_model_receiver);
        let game_updater =
            move |game_model: RLGameModel<W, H>, episode_rewards: Vec<f32>| -> bool {
                let model_clone = game_model.0.clone();
                let mut latest_model = latest_model_sender.lock().unwrap();
                *latest_model = Some((model_clone, episode_rewards));
                if let Ok(input) = main_to_agent_receiver.try_recv() {
                    match input {
                        GameInput::Quit => {
                            return true;
                        }
                        _ => return false,
                    }
                }
                false
            };
        let (train_info_tx, train_info_rx) = mpsc::channel();

        let max_iters = config.max_steps;

        let update_interval = 100;
        let agent_engine_handle = match model_type {
            ModelType::Burn => {
                let train_info_updater =
                    move |step: usize,
                          learning_rate: f32,
                          epsilon: f32,
                          loss_acc: Option<BurnTensor<Autodiff<Wgpu>, 4>>,
                          loss: BurnTensor<Autodiff<Wgpu>, 4>|
                          -> Option<BurnTensor<Autodiff<Wgpu>, 4>> {
                        let loss_acc = match loss_acc {
                            Some(loss_acc) => BurnTensor(loss_acc.0 + loss.0.detach()),
                            None => {
                                let inner_tensor = Tensor::zeros_like(&loss.0) + loss.0.detach();
                                BurnTensor(inner_tensor)
                            }
                        };
                        if step % update_interval == 0 {
                            let loss_value = (loss_acc.0 / update_interval as u32).into_scalar();
                            let _ = train_info_tx.send((loss_value, learning_rate, epsilon));
                            None
                        } else {
                            Some(loss_acc)
                        }
                    };

                thread::spawn(move || {
                    let backend = WgpuDevice::default();
                    let mut agent_engine = DQNet::new(backend);

                    let config_optimizer = AdamWConfig::new()
                        .with_weight_decay(1e-2)
                        .with_grad_clipping(Some(
                            burn::grad_clipping::GradientClippingConfig::Norm(1.0),
                        ));
                    let optimizer =
                        config_optimizer.init::<Autodiff<Wgpu>, Model<Autodiff<Wgpu>>>();

                    let _ = agent_engine.train_loop::<_, _, _, Model<Autodiff<Wgpu>>, _, _>(
                        game_updater,
                        train_info_updater,
                        config,
                        optimizer,
                        schedule,
                    );
                })
            }
            ModelType::Awesom => {
                let train_info_updater =
                    move |step: usize,
                          learning_rate: f32,
                          epsilon: f32,
                          loss_acc: Option<nn::tensor::Tensor<WgpuBackend>>,
                          mut loss: nn::tensor::Tensor<WgpuBackend>|
                          -> Option<nn::tensor::Tensor<WgpuBackend>> {
                        let loss = nn::tensor::Tensor::detach(&mut loss);
                        let loss_acc = match loss_acc {
                            Some(mut loss_acc) => {
                                let loss_acc = nn::tensor::Tensor::detach(&mut loss_acc);
                                nn::tensor::Tensor::add(&loss_acc, &loss).unwrap()
                            }
                            None => loss,
                        };
                        if step % update_interval == 0 {
                            let scalar_tensor = nn::tensor::Tensor::from_values(
                                loss_acc.backend().clone(),
                                vec![update_interval as f32],
                                Dimensions4D::new(1, 1, 1, 1),
                                false,
                            )
                            .unwrap();
                            let loss_value = nn::tensor::Tensor::div(&loss_acc, &scalar_tensor)
                                .unwrap()
                                .to_values()[0];
                            let _ = train_info_tx.send((loss_value, learning_rate, epsilon));
                            None
                        } else {
                            Some(loss_acc)
                        }
                    };

                thread::spawn(move || {
                    let backend = Rc::new(RefCell::new(WgpuBackend::default()));
                    let mut agent_engine = DQNet::new(backend.clone());

                    let optimizer =
                        AdamOptimizer::new(backend, 0.9, 0.999, 10e-8, Some(1e-2), Some(1.0))
                            .unwrap();

                    let _ = agent_engine.train_loop::<_, _, _, AwesomModel<WgpuBackend>, _, _>(
                        game_updater,
                        train_info_updater,
                        config,
                        optimizer,
                        schedule,
                    );
                })
            }
        };

        let agent_visual_model = GameModel::new((5, 5), 0);
        let agent_visual_info = Info::new(max_iters);
        Self {
            agent_engine_handle: Some(agent_engine_handle),
            latest_model_receiver,
            main_to_agent_sender,
            train_info_rx,
            agent_visual_model,
            agent_visual_info,
            update_interval,
        }
    }
}

impl<const W: usize, const H: usize> Engine for TrainingEngine<W, H> {
    fn tick(&mut self, user_input: Option<GameInput>) -> io::Result<bool> {
        let mut should_quit = false;
        if let Some(input) = user_input {
            let _ = self.main_to_agent_sender.send(input.clone());
            if let GameInput::Quit = input {
                should_quit = true
            }
        }
        if should_quit && self.agent_engine_handle.is_some() {
            let _ = self.main_to_agent_sender.send(GameInput::Quit);
            if let Some(agent_handle) = self.agent_engine_handle.take() {
                agent_handle.join().unwrap();
            }
        }
        {
            let mut guard = self.latest_model_receiver.lock().unwrap();
            if let Some((model, latest_rewards)) = guard.take() {
                self.agent_visual_model = model.clone();
                self.agent_visual_info.add_rewards(latest_rewards);
            }
        }
        while let Ok((train_info, lr, epsilon)) = self.train_info_rx.try_recv() {
            self.agent_visual_info
                .add_loss(self.update_interval, train_info);
            self.agent_visual_info.learning_rate = lr;
            self.agent_visual_info.exploration_rate = epsilon;
        }

        Ok(should_quit)
    }

    fn render_frame(&self, frame: &mut Frame) {
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(frame.size());
        frame.render_widget(&self.agent_visual_model, left);
        frame.render_widget(&self.agent_visual_info, right);
    }
}
