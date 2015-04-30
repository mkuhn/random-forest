#include <stdexcept>
#include <vector>
#include <random>

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"

inline void test(const bool& x) {
    if(!x) throw std::logic_error("test failed.");
}

template<class T>
void
printLiteral2D(andres::Marray<T> arr, const size_t shape[]) {
    std::cout << "{";
    for(size_t i = 0; i < shape[0]; ++i) {
        if (i) std::cout << ",";
        std::cout << "{";
        for(size_t j = 0; j < shape[1]; ++j) {
            if (j) std::cout << ",";
            std::cout << arr(i, j);
        }
        std::cout << "}";
    }
    std::cout << "}; " << std::endl;
}

int main() {
    const size_t numberOfSamples = 100;
    const size_t numberOfFeatures = 2;
    const size_t numberOfLabels = 2;
    
    // define random feature matrix
    std::default_random_engine RandomNumberGenerator;
    typedef double Feature;
    std::uniform_real_distribution<double> randomDistribution(0.0, 1.0);
    const size_t shape[] = {numberOfSamples, numberOfFeatures};
    const size_t label_shape[] = {numberOfSamples, numberOfLabels};
    andres::Marray<Feature> features(shape, shape + 2);
    for(size_t sample = 0; sample < numberOfSamples; ++sample)
    for(size_t feature = 0; feature < numberOfFeatures; ++feature) {
        features(sample, feature) = randomDistribution(RandomNumberGenerator);
    }

    // define labels
    typedef unsigned char Label;
    andres::Marray<Label> labels(shape, shape + 1);
    for(size_t sample = 0; sample < numberOfSamples; ++sample) {
        if((features(sample, 0) <= 0.5 && features(sample, 1) <= 0.5)
        || (features(sample, 0) > 0.5 && features(sample, 1) > 0.5)) {
            labels(sample) = 0;
        }
        else {
            labels(sample) = 1;
        }
    }

    // learn decision forest
    typedef double Probability;
    andres::ml::DecisionForest<Feature, Label, Probability> decisionForest;
    const size_t numberOfDecisionTrees = 10;
    decisionForest.learn(features, labels, numberOfLabels, numberOfDecisionTrees);

    // predict probabilities for every label and every training sample
    andres::Marray<Probability> probabilities(label_shape, label_shape + 2);
    decisionForest.predict(features, probabilities);
    // TODO: test formally

    // printLiteral2D(probabilities, label_shape);
    
    const Probability reference[numberOfSamples][numberOfFeatures] = {{0.1,0.9},{0.8,0.2},{1,0},{0.9,0.1},{0.9,0.1},{0.1,0.9},{0.9,0.1},{0.8,0.2},{1,0},{0,1},{0.2,0.8},{0.4,0.6},{0.6,0.4},{0.4,0.6},{0.2,0.8},{0.9,0.1},{1,0},{0.7,0.3},{0,1},{1,0},{1,0},{1,0},{0.1,0.9},{0.8,0.2},{0.2,0.8},{0.1,0.9},{0.2,0.8},{0.9,0.1},{0.9,0.1},{0.7,0.3},{0.4,0.6},{1,0},{0.3,0.7},{0.3,0.7},{0.9,0.1},{0.3,0.7},{0.3,0.7},{0.7,0.3},{0.7,0.3},{0.9,0.1},{0,1},{0.2,0.8},{0.8,0.2},{0.9,0.1},{0.7,0.3},{1,0},{0.1,0.9},{0,1},{0.7,0.3},{0.9,0.1},{1,0},{0.2,0.8},{0.9,0.1},{0.1,0.9},{0.7,0.3},{0.8,0.2},{0.1,0.9},{0.9,0.1},{1,0},{0.8,0.2},{0,1},{0.8,0.2},{0.9,0.1},{0.6,0.4},{0.8,0.2},{0.9,0.1},{0.3,0.7},{0.6,0.4},{0.2,0.8},{0.7,0.3},{0.3,0.7},{0.2,0.8},{0.8,0.2},{0.2,0.8},{0.1,0.9},{0.8,0.2},{0.7,0.3},{0.1,0.9},{0.1,0.9},{0.4,0.6},{0.1,0.9},{0.8,0.2},{0.2,0.8},{0.8,0.2},{0.7,0.3},{0.8,0.2},{0.4,0.6},{0.9,0.1},{0.2,0.8},{1,0},{0.2,0.8},{0.9,0.1},{0.8,0.2},{0.6,0.4},{0.1,0.9},{0.9,0.1},{0.9,0.1},{1,0},{0.2,0.8},{1,0}};

    for(size_t i = 0; i < label_shape[0]; ++i) {
        for(size_t j = 0; j < label_shape[1]; ++j) {
            test( probabilities(i, j) == reference[i][j] );
        }
    }

    
    
    return 0;
}
