use super::environment::{RLEnvironment, Replay};
use rand::seq::IteratorRandom;
use std::collections::VecDeque;

pub struct ReplayBuffer<E: RLEnvironment> {
    memory: VecDeque<Replay<E>>,
    max_size: usize,
}

// TODO: implement this as a ring buffer
impl<E: RLEnvironment> ReplayBuffer<E> {
    pub fn new(max_size: usize) -> Self {
        Self {
            memory: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add(&mut self, replay: Replay<E>) {
        if self.memory.len() >= self.max_size {
            self.memory.pop_front();
        }
        self.memory.push_back(replay);
    }

    pub fn get(&self, amount: usize) -> Vec<&Replay<E>> {
        let mut rng = rand::thread_rng();
        let (front, back) = self.memory.as_slices();
        front
            .iter()
            .chain(back.iter())
            .choose_multiple(&mut rng, amount)
    }
}
