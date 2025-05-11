use std::io::{self};

use crossterm::event::{self, Event, KeyCode};

#[derive(Clone, Debug)]
pub enum GameInput {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Quit,
    NextLevel,
}

// fn handle_input() -> io::Result<GameInput> {}

pub fn handle_events() -> io::Result<Option<GameInput>> {
    if event::poll(std::time::Duration::from_millis(50))? {
        if let Event::Key(key) = event::read()? {
            if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                return Ok(Some(GameInput::Quit));
            } else if key.code == KeyCode::Char('h')
                || key.code == KeyCode::Left
                || key.code == KeyCode::Char('a')
            {
                return Ok(Some(GameInput::MoveLeft));
            } else if key.code == KeyCode::Char('j')
                || key.code == KeyCode::Down
                || key.code == KeyCode::Char('s')
            {
                return Ok(Some(GameInput::MoveDown));
            } else if key.code == KeyCode::Char('k')
                || key.code == KeyCode::Up
                || key.code == KeyCode::Char('w')
            {
                return Ok(Some(GameInput::MoveUp));
            } else if key.code == KeyCode::Char('l')
                || key.code == KeyCode::Right
                || key.code == KeyCode::Char('d')
            {
                return Ok(Some(GameInput::MoveRight));
            } else if key.code == KeyCode::Char('n') {
                return Ok(Some(GameInput::NextLevel));
            }
        }
    }
    Ok(None)
}
