
public class RecurrentNN implements Network, Cloneable{

	Neuron[] inputLayer;
	Neuron[] hiddenInputLayer; //copy of values of hiddenLayer
	Neuron[] pseudoInputLayer; //inputLayer + hiddenInputLayer
	Neuron[] hiddenLayer;
	Neuron[] outputLayer;
	float learningRate = .01f;
	float momentum = .9f;
	boolean confidence;
	
	public RecurrentNN(int inputSize, int hiddenSize, int outputSize){
		inputLayer = new Neuron[inputSize];
		hiddenInputLayer = new Neuron[hiddenSize];
		pseudoInputLayer = new Neuron[inputSize + hiddenSize];
		hiddenLayer = new Neuron[hiddenSize];
		outputLayer = new Neuron[outputSize];
		
		for(int i = 0; i < inputSize; i++){
			inputLayer[i] = new Neuron();
			pseudoInputLayer[i] = inputLayer[i];
		}
		for(int i = 0; i < hiddenSize; i++){
			hiddenInputLayer[i] = new Neuron();
			pseudoInputLayer[inputSize+i] = hiddenInputLayer[i];
			hiddenLayer[i] = new Neuron();
		}
		for(int i = 0; i < outputSize; i++){
			outputLayer[i] = new Neuron();
		}
		randomize();
	}
	
	public RecurrentNN(int inputSize, int hiddenSize, int outputSize, float[][] weights){
		inputLayer = new Neuron[inputSize];
		hiddenInputLayer = new Neuron[hiddenSize];
		pseudoInputLayer = new Neuron[inputSize + hiddenSize];
		hiddenLayer = new Neuron[hiddenSize];
		outputLayer = new Neuron[outputSize];
		
		for(int i = 0; i < inputSize; i++){
			inputLayer[i] = new Neuron();
			pseudoInputLayer[i] = inputLayer[i];
		}
		for(int i = 0; i < hiddenSize; i++){
			hiddenInputLayer[i] = new Neuron();
			pseudoInputLayer[inputSize+i] = hiddenInputLayer[i];
			hiddenLayer[i] = new Neuron();
		}
		for(int i = 0; i < outputSize; i++){
			outputLayer[i] = new Neuron();
		}
		for(int i = 0; i < hiddenSize; i++){
			setNodeWeight(weights[i]);
		}
	}
	
	public int getInputLength(){
		return inputLayer.length;
	}
	
	public void clear(){
		for(int i = 0; i < pseudoInputLayer.length; i++){
			pseudoInputLayer[i].outputWeight.clear();
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			hiddenLayer[i].outputWeight.clear();
		}
	}
	
	void randomize(){
		for(int i = 0; i < pseudoInputLayer.length; i++){
			for(int j = 0; j < hiddenLayer.length; j++){
				pseudoInputLayer[i].setOutputWeight(hiddenLayer[j].getId(), (float) (Math.random() * 2) - 1);
			}
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			for(int j = 0; j < outputLayer.length; j++){
				hiddenLayer[i].setOutputWeight(outputLayer[j].getId(), (float) (Math.random() * 2) - 1);
			}
		}
	}
	
	public float[] run(float[] inputs){
		float[] outputs = new float[outputLayer.length];
		
		for(int i = 0; i < inputLayer.length; i++){
			inputLayer[i].setValue(inputs[i]);
		}
		for(int i = 0; i < hiddenInputLayer.length; i++){
			hiddenInputLayer[i].setValue(hiddenLayer[i].getValue());
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			hiddenLayer[i].calcValue(pseudoInputLayer);
		}
		for(int i = 0; i < outputLayer.length; i++){
			outputs[i] = outputLayer[i].calcValue(hiddenLayer);
		}
		return outputs;
	}
	
	public void setNodeWeight(float[] weights){
		//size of array must equal input plus hidden plus output
		int hiddenNode = -1;
		for(int i = 0; i < hiddenLayer.length; i++){
			if(hiddenLayer[i].outputWeight.isEmpty()){
				hiddenNode = i;
				break;
			}
		}
		if(hiddenNode == -1){
			return;
		}
		for(int i = 0; i < weights.length; i++){
			if(i < pseudoInputLayer.length){
				pseudoInputLayer[i].setOutputWeight(hiddenLayer[hiddenNode].getId(), weights[i]);
			}else{
				hiddenLayer[hiddenNode].setOutputWeight(outputLayer[i-pseudoInputLayer.length].getId(), weights[i]);
			}
		}
	}
	
	/*public void addHidden(){
		Neuron[] hiddenLayer = new Neuron[this.hiddenLayer.length + 1];
		for(int i = 0; i < this.hiddenLayer.length; i++){
			hiddenLayer[i] = this.hiddenLayer[i].clone();
		}
		hiddenLayer[hiddenLayer.length - 1] = new Neuron();
		int[] outputArray = new int[outputLayer.length];
		for(int i = 0; i < outputLayer.length; i++){
			outputArray[i] = outputLayer[i].getId();
		}
		hiddenLayer[hiddenLayer.length - 1].randomOutputWeight(outputArray);
		for(int i = 0; i < inputLayer.length; i++){
			inputLayer[i].randomOutputWeight(hiddenLayer[hiddenLayer.length - 1].getId());
		}
		this.hiddenLayer = Arrays.copyOf(hiddenLayer, hiddenLayer.length);
	}
	public void deleteHidden(int id){
		int location = -1;
		for(int i = 0; i < hiddenLayer.length; i++){
			if(id == hiddenLayer[i].getId()){
				location = i;
			}
		}
		if(location != -1){
			Neuron[] hiddenLayer = new Neuron[this.hiddenLayer.length - 1];
			int count = 0;
			for(int i = 0; i < this.hiddenLayer.length; i++){
				if(i != location){
					hiddenLayer[count] = this.hiddenLayer[i].clone();
					count++;
				}
			}
			for(int i = 0; i < inputLayer.length; i++){
				inputLayer[i].outputWeight.remove(id);
			}
			this.hiddenLayer = Arrays.copyOf(hiddenLayer, hiddenLayer.length);
		}
	}*/
	
	
	
	
	double[] runLayer(Neuron[] inputLayer, Neuron[] layer){
		double[] outputs = new double[layer.length];
		
		for(int i = 0; i < layer.length; i++){
			layer[i].calcValue(inputLayer);
		}
		return outputs;
	}
	
	public void backpropagation(float[][] trainIn, float[][] trainOut){
		float[][] actualOut = new float[trainIn.length][];
		double[] error;
		double[] hiddenError;
		double[][] deltaWeight = new double[outputLayer.length][hiddenLayer.length];
		double[][] deltaHiddenWeight = new double[hiddenLayer.length][pseudoInputLayer.length];
		double[][] oldDeltaWeight = new double[outputLayer.length][hiddenLayer.length];
		double[][] oldDeltaHiddenWeight = new double[hiddenLayer.length][pseudoInputLayer.length];
		for(int i = 0; i < oldDeltaWeight.length; i++){
			for(int j = 0; j < oldDeltaWeight[i].length; j++){
				oldDeltaWeight[i][j] = 0;
			}
		}
		for(int i = 0; i < oldDeltaHiddenWeight.length; i++){
			for(int j = 0; j < oldDeltaHiddenWeight.length; j++){
				oldDeltaHiddenWeight[i][j] = 0;
			}
		}
		//for each training set
		for(int i = 0; i < actualOut.length; i++){
			//run training set
			actualOut[i] = run(trainIn[i]);
			//find error
			error = new double[actualOut[i].length];
			//for each output node
			for(int j = 0; j < error.length; j++){
				//difference times derivative of activation function
				error[j] = (trainOut[i][j] - actualOut[i][j]) * (1 - Math.pow(Math.tanh(actualOut[i][j]), 2));
				//backpropagate error
				//change inputs into output nodes
				for(int k = 0; k < hiddenLayer.length; k++){
					deltaWeight[j][k] = learningRate * error[j] * hiddenLayer[k].getOutput() + oldDeltaWeight[j][k] * momentum;
					hiddenLayer[k].setOutputWeight(outputLayer[j].getId(), (float) (deltaWeight[j][k] + hiddenLayer[k].getOutputWeight(outputLayer[j].getId())));
					oldDeltaWeight[j][k] = deltaWeight[j][k];
				}
			
			}
			//for each hidden node
			hiddenError = new double[hiddenLayer.length];
			for(int j = 0; j < hiddenLayer.length; j++){
				double sumError = 0;
				for(int k = 0; k < error.length; k++){
					sumError += error[k] * hiddenLayer[j].getOutputWeight(outputLayer[k].getId());
				}
				hiddenError[j] = (1 - Math.pow(hiddenLayer[j].getOutput(), 2)) * sumError;
				for(int k = 0; k < pseudoInputLayer.length; k++){
					deltaHiddenWeight[j][k] = learningRate * hiddenError[j] * pseudoInputLayer[k].getOutput() + oldDeltaHiddenWeight[j][k] * momentum;
					pseudoInputLayer[k].setOutputWeight(hiddenLayer[j].getId(), (float) (deltaHiddenWeight[j][k] + pseudoInputLayer[k].getOutputWeight(hiddenLayer[j].getId())));
					oldDeltaHiddenWeight[j][k] = deltaHiddenWeight[j][k];
				}
			}
			//calculate global error
		}
	}
	
	public RecurrentNN clone(){
		try {
			RecurrentNN clone = (RecurrentNN) super.clone();
			for(int i = 0; i < inputLayer.length; i++){
				clone.inputLayer[i] = inputLayer[i].clone();
				clone.pseudoInputLayer[i] = clone.inputLayer[i];
			}
			for(int i = 0; i < hiddenLayer.length; i++){
				clone.hiddenLayer[i] = hiddenLayer[i].clone();
			}
			for(int i = 0; i < hiddenInputLayer.length; i++){
				clone.hiddenInputLayer[i] = hiddenInputLayer[i].clone();
				clone.pseudoInputLayer[i-inputLayer.length] = clone.hiddenInputLayer[i]; 
			}
			for(int i = 0; i < outputLayer.length; i++){
				clone.outputLayer[i] = outputLayer[i].clone();
			}
			return clone;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void setConfidence(boolean confidence){
		this.confidence = confidence;
	}
	
	public boolean getConfidence(){
		return confidence;
	}

}
