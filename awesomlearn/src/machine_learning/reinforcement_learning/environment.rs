use crate::machine_learning::nn::dimensions::Dimensions3D;

use std::fmt::Debug;

// envrionment.rs
pub type Reward = f32;

pub struct Replay<E: RLEnvironment> {
    pub prev_state: E::State,
    pub action: E::Action,
    pub reward: Reward,
    pub next_state: E::State,
    pub terminal: bool,
}
pub type Step<E> = (
    <E as RLEnvironment>::Action,
    Reward,
    <E as RLEnvironment>::State,
    bool,
);

pub trait RLAction:
    Clone + Debug + Copy + Into<f32> + From<f32> + From<u32> + Into<u32> + Sync + Send
{
    const SIZE: u32;
}

pub trait RLState: Clone + Debug + AsRef<[f32]> + Sync + Send {
    const SIZE: usize;
}

pub trait RLEnvironment: Sync + Send + 'static {
    type State: RLState;
    type Action: RLAction;

    fn init() -> Self;
    fn record_step(&mut self, action: Self::Action) -> Step<Self> {
        // let prev_state = self.state();
        let (reward, ended) = self.step(&action);
        let next_state = self.state();
        (action, reward, next_state, ended)
    }
    fn cumulative_reward(&self) -> Reward;
    fn step(&mut self, action: &Self::Action) -> (Reward, bool);
    fn reset(&mut self);
    fn state(&self) -> Self::State;
    fn dimensions(&self) -> Dimensions3D;
}

pub trait RLCurriculum {
    fn set_difficulty(&self, level: usize);
}
