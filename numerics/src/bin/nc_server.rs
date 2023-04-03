use node_crunch::*;

struct NcServer {
    counter: u32,
}

impl NCServer for NcServer {
    fn prepare_data_for_node(&mut self, node_id: NodeID) -> Result<NCJobStatus, NCError> {
        todo!()
    }

    fn process_data_from_node(&mut self, node_id: NodeID, data: &[u8]) -> Result<(), NCError> {
        todo!()
    }

    fn heartbeat_timeout(&mut self, nodes: Vec<NodeID>) {
        todo!()
    }

    fn finish_job(&mut self) {
        println!("success!");
    }


}

fn main() {
    
}
