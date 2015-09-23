import java.util.Arrays;



public class Genetics {

	Chromosome[] population;
	static final int popSize = 100;
	static final int keepHigh = 25; //percentage of highest to keep; rest of them are killed after crossover
	int generation;
	static int trial = 3;
	
	/*
	public Genetics(){
		population = new Chromosome[popSize];
		for(int i = 0; i < popSize; i++){
			population[i] = new Chromosome();
		}
	}
	*/
	
	public static Genetics importGenetics(Chromosome[] populationImport, int generationImport){
		Genetics genetics = new Genetics();
		genetics.population = new Chromosome[populationImport.length];
		for(int i = 0; i < populationImport.length; i++){
			genetics.population[i] = populationImport[i].clone();
		}
		genetics.generation = generationImport;
		genetics.newPop();
		return genetics;
	}
	
	public static Genetics importGenetics(String[] networks, int gen){
		Genetics genetics = new Genetics();
		genetics.population = new Chromosome[networks.length];
		for(int i = 0; i < networks.length; i++){
			String[] networkInfo = networks[i].split("(;| )");
			int inputLength = Integer.parseInt(networkInfo[1]);
			int hiddenLength = Integer.parseInt(networkInfo[2]);
			int outputLength = Integer.parseInt(networkInfo[3]);
			float[][] weightMap = new float[hiddenLength][inputLength + outputLength];
			for(int j = 4; j < networkInfo.length; j++){
				weightMap[(j-4)/(inputLength+outputLength)][(j-4)%(inputLength+outputLength)] = Float.parseFloat(networkInfo[j]);
			}
			genetics.population[i] = genetics.new Chromosome(inputLength, hiddenLength, outputLength, weightMap);
		}
		genetics.generation = gen;
		genetics.newPop();
		return genetics;
	}
	
	public Genetics(){
		
	}
	
	public Genetics(int inputSize, int hiddenSize, int outputSize){
		population = new Chromosome[popSize];
		for(int i = 0; i < popSize; i++){
			population[i] = new Chromosome(inputSize, hiddenSize, outputSize);
		}
		generation = 1;
	}
	
	private int[] add(int[] array, int value, int location){
		int[] newarray = new int[array.length];
		for(int i = 0; i < location; i++){
			newarray[i] = array[i];
		}
		for(int i = array.length - 1; i > location; i--){
			newarray[i] = array[i-1];
		}
		newarray[location] = value;
		return newarray;
	}
	
	public void newPop(){
		Chromosome[] newPopulation = new Chromosome[popSize];
		int[] fitnessOrder = new int[popSize];
		int count = 0;
		for(int i = 0; i < popSize; i++){
			if(i == 0){
				fitnessOrder[0] = 0;
				count++;
			}else{
				for(int j = 0; j < count; j++){
					if(population[i].fitness > population[fitnessOrder[j]].fitness){
						fitnessOrder = add(fitnessOrder, i, j);
						break;
					}else if(j == count - 1){
						fitnessOrder[i] = i;
					}
				}
			}
		}
		for(int i = 0; i < popSize/(100/keepHigh); i++){
			newPopulation[i] = population[fitnessOrder[i]].clone();
		}
		for(int i = popSize/(100/keepHigh); i < popSize; i++){
			newPopulation[i] = crossover();
		}
		population = new Chromosome[popSize];
		population = newPopulation.clone();
	}
	
	Chromosome crossover(){
		Chromosome father = chooseFather();
		Chromosome mother = chooseMother(Arrays.asList(population).indexOf(father));
		Chromosome child;
		int fatherHiddenSize = father.network.hiddenLayer.length;
		int motherHiddenSize = mother.network.hiddenLayer.length;
		int lowerHiddenSize;
		if(fatherHiddenSize <= motherHiddenSize){
			lowerHiddenSize = fatherHiddenSize;
		}else{
			lowerHiddenSize = motherHiddenSize;
		}
		int childHiddenSize = (int) (Math.random() * (Math.abs(fatherHiddenSize - motherHiddenSize)) + lowerHiddenSize);
		
		child = new Chromosome(father.network.inputLayer.length, childHiddenSize, father.network.outputLayer.length);
		child.network.clear();
		//choose neuron from either father or mother
		//put weights into array
		float[][] childWeights = new float[childHiddenSize][child.network.inputLayer.length + child.network.outputLayer.length];
		for(int i = 0; i < childHiddenSize; i++){
			//choose parent
			int parent = (int) (Math.random() * 2);
			if(parent == 0){
				//from father
				int neuron = (int) (Math.random() * father.network.hiddenLayer.length);
				for(int j = 0; j < father.network.inputLayer.length; j++){
					childWeights[i][j] = father.network.inputLayer[j].outputWeight.get(father.network.hiddenLayer[neuron].getId());
				}
				for(int j = 0; j < father.network.outputLayer.length; j++){
					childWeights[i][j+father.network.inputLayer.length] = father.network.hiddenLayer[neuron].outputWeight.get(father.network.outputLayer[j].getId());
				}
			}else{
				//from mother
				int neuron = (int) (Math.random() * mother.network.hiddenLayer.length);
				for(int j = 0; j < mother.network.inputLayer.length; j++){
					childWeights[i][j] = mother.network.inputLayer[j].outputWeight.get(mother.network.hiddenLayer[neuron].getId());
				}
				for(int j = 0; j < mother.network.outputLayer.length; j++){
					childWeights[i][j+mother.network.inputLayer.length] = mother.network.hiddenLayer[neuron].outputWeight.get(mother.network.outputLayer[j].getId());
				}
			}
		}
		//insert neuron weights into child
		for(int i = 0; i < childWeights.length; i++){
			child.setNodeWeight(childWeights[i]);
		}
		mutation(child);
		return child.clone();
	}
	
	void mutation(Chromosome mutant){
		//chance for extra hidden node
		//int gainNeuron = 5;
		//chance for losing hidden node
		//int loseNeuron = 5;
		//chance for a given weight to be random
		int randomWeight = 5;
		//note: this seems like it'd be far too slow.  Look for ways to speed it up.
		
		double chance = Math.random() * 100;
		
		//if(chance < gainNeuron){
		//	mutant.addHidden();
		//}else if(chance < gainNeuron + loseNeuron){
		//	if(mutant.network.hiddenLayer.length > 1){
		//		mutant.subtractHidden();
		//	}
		//}
		for(int i = 0; i < mutant.network.inputLayer.length; i++){
			for(int j = 0; j < mutant.network.inputLayer[i].outputWeight.keySet().size(); j++){
				chance = Math.random() * 100;
				if(chance < randomWeight){
					mutant.network.inputLayer[i].outputWeight.replace((int) mutant.network.inputLayer[i].outputWeight.keySet().toArray()[j], (float) (Math.random() * 2) - 1);
				}
			}
		}
		for(int i = 0; i < mutant.network.hiddenLayer.length; i++){
			for(int j = 0; j < mutant.network.hiddenLayer[i].outputWeight.keySet().size(); j++){
				chance = Math.random() * 100;
				if(chance < randomWeight){
					mutant.network.hiddenLayer[i].outputWeight.replace((int) mutant.network.hiddenLayer[i].outputWeight.keySet().toArray()[j], (float) (Math.random() * 2) - 1);
				}
			}
		}
	}
	
	Chromosome chooseFather(){
		int totalFitness = 0;
		Chromosome father = null;
		for(int i = 0; i < population.length; i++){
			totalFitness += population[i].fitness;
		}
		int parent1 = (int) (Math.random() * totalFitness);
		for(int i = 0; i < population.length; i++){
			if(parent1 <= population[i].fitness){
				father = population[i];
				break;
			}else{
				parent1 -= population[i].fitness;
			}
		}
		return father;
	}
	
	Chromosome chooseMother(int fatherLocation){
		int totalFitness = 0;
		Chromosome mother = null;
		for(int i = 0; i < population.length; i++){
			if(i != fatherLocation){
				totalFitness += population[i].fitness;
			}
		}
		int parent2 = (int) (Math.random() * totalFitness);
		for(int i = 0; i < population.length; i++){
			if(i != fatherLocation){
				if(parent2 <= population[i].fitness){
					mother = population[i];
					break;
				}else{
					parent2 -= population[i].fitness;
				}
			}
		}
		return mother;
	}
	void chooseParents(Chromosome father, Chromosome mother){
		int totalFitness = 0;
		for(int i = 0; i < population.length; i++){
			totalFitness += population[i].fitness;
		}
		int parent1 = (int) (Math.random() * totalFitness);
		int fatherNumber = -1;
		for(int i = 0; i < population.length; i++){
			if(parent1 <= population[i].fitness){
				father = population[i];
				fatherNumber = i;
				totalFitness -= population[i].fitness;
				break;
			}else{
				parent1 -= population[i].fitness;
			}
		}
		int parent2 = (int) (Math.random() * totalFitness);
		for(int i = 0; i < population.length; i++){
			if(i != fatherNumber){
				if(parent2 <= population[i].fitness){
					mother = population[i];
					break;
				}else{
					parent2 -= population[i].fitness;
				}
			}
		}
	}
	
	public class Chromosome implements Cloneable{
		
		BasicNN network;
		Account account;
		static final int DEFAULT_INPUT_SIZE = 3;
		static final int DEFAULT_HIDDEN_SIZE = 3;
		static final int DEFAULT_OUTPUT_SIZE = 1;
		int fitness;
		int brokeCount;
		
		/*
		Chromosome(){
			network = new BasicNN(DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_OUTPUT_SIZE);
		}
		Chromosome(int hiddenSize){
			network = new BasicNN(DEFAULT_INPUT_SIZE, hiddenSize, DEFAULT_OUTPUT_SIZE);
		}
		*/
		Chromosome(int inputSize, int hiddenSize, int outputSize){
			network = new BasicNN(inputSize, hiddenSize, outputSize);
			brokeCount = 0;
			account = new Account();
		}
		Chromosome(int inputSize, int hiddenSize, int outputSize, float[][] weightMap){
			network = new BasicNN(inputSize, hiddenSize, outputSize, weightMap);
			brokeCount = 0;
			account = new Account();
		}
		void changeRandomWeight(){
			float weight = (float) (Math.random() * 2) - 1;
			changeRandomWeight(weight);
		}
		void changeRandomWeight(float weight){
			int totalInputWeights = network.inputLayer.length * network.hiddenLayer.length;
			int totalHiddenWeights = network.hiddenLayer.length * network.outputLayer.length;
			int weightNumber = totalInputWeights + totalHiddenWeights;
			int weightChoice = (int) Math.random() * weightNumber + 1;
			if(weightChoice < totalInputWeights){
				int inputChoice = weightChoice / network.hiddenLayer.length;
				int outputChoice = weightChoice % network.hiddenLayer.length;
				network.inputLayer[inputChoice].setOutputWeight(network.hiddenLayer[outputChoice].getId(), weight);
			}else{
				int inputChoice = weightChoice / network.outputLayer.length;
				int outputChoice = weightChoice % network.outputLayer.length;
				network.hiddenLayer[inputChoice].setOutputWeight(network.outputLayer[outputChoice].getId(), weight);
			}
			
		}
		void addHidden(){
			network.addHidden();
		}
		void setNodeWeight(float[] weights){
			network.setNodeWeight(weights);
		}
		void subtractHidden(){
			int neuron = (int) (Math.random() * network.hiddenLayer.length);
			int id = network.hiddenLayer[neuron].getId();
			network.deleteHidden(id);
		}
		
		public Chromosome clone(){
			try{
				Chromosome clone = (Chromosome) super.clone();
				clone.brokeCount = 0;
				clone.account = new Account();
				clone.network = network.clone();
				return clone;
			}catch(Exception e){
				e.printStackTrace();
			}
			return null;
		}
		
		
	}
	
	public String outputPop(){
		String output = "" + population.length;
		
		for(int i = 0; i < population.length; i++){
			output += "\n" + population[i].fitness + ";" + IO.output(population[i].network);
		}
		
		return output;
	}
	
	public void finished(){
		String filename = "Trial" + trial + "Generation" + generation + ".gene";
		IO.output(this, filename);
		generation++;
		this.newPop();
	}
}
