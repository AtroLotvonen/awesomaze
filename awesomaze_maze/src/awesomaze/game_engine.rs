// game_engine.rs

use ratatui::Frame;
use std::io;

use super::{
    engine::Engine,
    game_model::GameModel,
    input::GameInput,
};

pub struct GameEngine {
    pub game_model: GameModel,
    maze_x: Box<dyn Iterator<Item = usize>>,
    maze_y: Box<dyn Iterator<Item = usize>>,
    level: Box<dyn Iterator<Item = u32>>,
}

impl GameEngine {
    pub fn new() -> GameEngine {
        let mut maze_x = (11..103).step_by(2);
        let mut maze_y = (11..63).step_by(2);

        let maze_size = (maze_x.next().unwrap_or(103), maze_y.next().unwrap_or(63));
        let mut level = 1..;
        let game_model = GameModel::new(maze_size, level.next().unwrap_or(100));
        GameEngine {
            game_model,
            maze_x: Box::new(maze_x),
            maze_y: Box::new(maze_y),
            level: Box::new(level),
        }
    }

    // pub fn game_model(&self) -> &GameModel {
    //     &self.game_model
    // }
}

impl Engine for GameEngine {
    fn tick(&mut self, user_input: Option<GameInput>) -> io::Result<bool> {
        let mut should_quit = false;
        // rendering
        match user_input {
            Some(GameInput::Quit) => should_quit = true,
            Some(user_input) => self.game_model.handle_input(user_input),
            _ => {}
        }
        if self.game_model.has_game_ended() {
            let maze_size = (
                self.maze_x.next().unwrap_or(103),
                self.maze_y.next().unwrap_or(63),
            );
            self.game_model = GameModel::new(maze_size, self.level.next().unwrap_or(100));
        }
        Ok(should_quit)
    }

    fn render_frame(&self, frame: &mut Frame) {
        frame.render_widget(&self.game_model, frame.size());
    }
}
