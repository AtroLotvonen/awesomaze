use std::rc::Rc;

#[derive(Debug)]
pub struct WgpuScratchStorage {
    buffers: Vec<Rc<wgpu::Buffer>>,
}

// TODO: wrapper for scracth buffer so the rc cant be cloned elsewhere or fiddled?
// #[derive(Debug)]
// pub struct ScratchBuffer(Rc<wgpu::Buffer>);

impl WgpuScratchStorage {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    pub fn get_scratch_buffer(&mut self, min_size: u64) -> Option<Rc<wgpu::Buffer>> {
        // If available buffers return clone
        match self.get_smallest_available_buffer_index(min_size) {
            Some(i) => Some(self.buffers[i].clone()),
            None => None,
        }
    }

    /// Creates a new buffer, adds the created buffer to the list, removes the smallest not used buffer
    /// from the list and returns a clone of the created
    pub fn add_buffer(&mut self, new_scratch: wgpu::Buffer) -> Rc<wgpu::Buffer> {
        // first remove the smallest unnecessary buffer, so that the new created buffer isn't
        // removed. Helps to keep the scratch space as small as necessary
        if let Some(i) = self.get_smallest_available_buffer_index(0) {
            let _ = self.buffers.remove(i);
        }
        // find the partition point where to insert the new buffer
        let partition_point = self
            .buffers
            .partition_point(|x| x.size() >= new_scratch.size());
        let buffer = Rc::new(new_scratch);
        self.buffers.insert(partition_point, buffer.clone());
        buffer
    }

    /// Return the smallest available buffer index specified by the min_size. None if not any.
    fn get_smallest_available_buffer_index(&self, min_size: u64) -> Option<usize> {
        let partition_point = self.buffers.partition_point(|x| x.size() >= min_size);
        // if the partition point is at the beginning, all buffers are too small
        if partition_point == 0 {
            return None;
        }
        // TODO: would it be better to remove the first largest buffer size here?
        let candidate = self.buffers[0..partition_point]
            .iter()
            .enumerate()
            // find the smallest which isnt already in use by the rc count so reverse and check
            .rev()
            .find(|(_i, scratch)| Rc::strong_count(scratch) == 1)
            .as_ref()
            // Return the index
            .map(|(i, _)| *i);
        candidate
    }
}

impl Default for WgpuScratchStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::nn::backends::wgpu_backend::WgpuBackend;

    use super::*;

    fn get_scratch(
        backend: &mut WgpuBackend,
        scratch_storage: &mut WgpuScratchStorage,
        min_size: u64,
    ) -> Rc<wgpu::Buffer> {
        match scratch_storage.get_scratch_buffer(min_size) {
            Some(buffer) => buffer,
            None => {
                let buffer = backend.create_buffer(min_size);
                scratch_storage.add_buffer(buffer)
            }
        }
    }

    #[test]
    fn test_scratch_storage() {
        let mut backend = WgpuBackend::default();
        let (a_size, b_size, c_size) = (100, 200, 300);
        let mut scratch_storage = WgpuScratchStorage::default();
        {
            let _c = get_scratch(&mut backend, &mut scratch_storage, c_size);
        }
        assert_eq!(scratch_storage.buffers.len(), 1);
        assert_eq!(scratch_storage.buffers[0].size(), c_size);
        assert_eq!(Rc::strong_count(&scratch_storage.buffers[0]), 1);

        {
            let b = get_scratch(&mut backend, &mut scratch_storage, b_size);
            assert_eq!(b.size(), c_size);
            assert_eq!(scratch_storage.buffers.len(), 1);

            let a = get_scratch(&mut backend, &mut scratch_storage, a_size);

            assert_eq!(a.size(), a_size);
            assert_eq!(scratch_storage.buffers.len(), 2);
        }
        let d_size = 400;
        // Repeat the
        {
            // This should make a new buffer with a size d_size and remove the smallest which is
            // a_size
            let d = get_scratch(&mut backend, &mut scratch_storage, d_size);
            assert_eq!(d.size(), d_size);
            assert_eq!(scratch_storage.buffers.len(), 2);

            // TODO: would it be better to remove the first largest buffer size so the a would get
            // the b_size here?
            //
            // The a size buffer should be removed and the a buffer should have the c_size(???)
            let a = get_scratch(&mut backend, &mut scratch_storage, a_size);

            assert_eq!(a.size(), c_size);
            assert_eq!(scratch_storage.buffers.len(), 2);
        }
    }
}
