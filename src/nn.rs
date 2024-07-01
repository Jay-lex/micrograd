use crate::engine::Value;
use rand::Rng;
use std::iter::zip;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Neuron {
        Neuron {
            w: (0..nin).map(|_| Value::from(rand::thread_rng().gen_range(-1.0..=1.0))).collect(),
            b: Value::from(rand::thread_rng().gen_range(-1.0..=1.0)),
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        (zip(&self.w, x)
            .map(|(wi, xi)| wi * xi)
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|a, b| a + b)
            .unwrap())
            + self.b.clone().relu()
    }

    pub fn parameters(&self) -> Vec<Value> {
        [self.w.clone(), vec![self.b.clone()]].concat()
    }
}

impl std::fmt::Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // make it ReLU or Linear
        write!(f, "Linear Neuron({})", self.w.len())
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|i| i.forward(x)).collect::<Vec<_>>()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters()).collect()
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer of [{:?}]",
            self.neurons.iter().map(|n| format!("{:?}", n)).collect::<Vec<_>>().join(", ")
        )
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> MLP {
        MLP {
            layers: (0..nouts.len())
                .map(|i| Layer::new(vec![nin].into_iter().chain(nouts.clone().into_iter()).collect::<Vec<_>>()[i], nouts[i]))
                .collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.layers.iter().flat_map(|i| i.forward(&x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLP of [{}]",
            self.layers.iter().map(|l| format!("{:?}", l)).collect::<Vec<_>>().join(", ")
        )
    }
}
