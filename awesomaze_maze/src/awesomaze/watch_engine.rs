// game_engine.rs

use awesomlearn::machine_learning::reinforcement_learning::environment::RLEnvironment;
use ratatui::Frame;
use std::io;

use super::{engine::Engine, game_model::RLGameModel, input::GameInput, model::InferenceModel};

pub struct WatchEngine<const W: usize, const H: usize, I: InferenceModel> {
    rl_env: RLGameModel<W, H>,
    inference_model: I,
    steps_in_level: usize,
}

impl<const W: usize, const H: usize, I: InferenceModel> WatchEngine<W, H, I> {
    pub fn new(inference_model: I) -> Self {
        let rl_env = RLGameModel::<W, H>::init();
        Self {
            rl_env,
            inference_model,
            steps_in_level: 0,
        }
    }

    fn reset_random_level(&mut self) {
        self.rl_env.random_level();
        self.steps_in_level = 0;
    }
}

impl<const W: usize, const H: usize, I: InferenceModel> Engine for WatchEngine<W, H, I> {
    fn tick(&mut self, user_input: Option<GameInput>) -> io::Result<bool> {
        let mut should_quit = false;
        // rendering
        match user_input {
            Some(GameInput::Quit) => should_quit = true,
            Some(GameInput::NextLevel) => self.reset_random_level(),
            _ => {}
        }
        if self.rl_env.0.has_game_ended()
            || (self.rl_env.0.bounds().0 * self.rl_env.0.bounds().1) / 2 < self.steps_in_level
        {
            self.reset_random_level();
        }
        let state: Vec<f32> = self.rl_env.state().as_ref().to_vec();
        let action = self.inference_model.predict_move(state);
        let _ = self.rl_env.step(&action);
        self.steps_in_level += 1;
        Ok(should_quit)
    }

    fn render_frame(&self, frame: &mut Frame) {
        frame.render_widget(&self.rl_env.0, frame.size());
    }
}
