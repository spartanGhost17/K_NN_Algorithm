classdef myknn
    methods(Static)
        %% takes training examples, training labels and k (number of nearest neighbours)
        %% increase model performance by standardizing data (z-score standardization) to avoid usefull (discriminatory features) to be drowned out in later stages
        %% return trained model
        function m = fit(train_examples, train_labels, k)
            
            % start of standardisation process (increase model accuracy)
            %% calculate mean of all features in table
			m.mean = mean(train_examples{:,:});
            %% calculate standard deviation (how much feature differ from mean) of all features in table
			m.std = std(train_examples{:,:})
            %% for however many rows we have in our train example matrix apply z-score standardization to avoid individual large features drowning out those with smaller footprint in euclidean calculation
            for i=1:size(train_examples,1)
                %% substract mean from current training example to have feature centered at zero
				train_examples{i,:} = train_examples{i,:} - m.mean;
                %% scales down features with big spreads and scales up features with small spreads, all feature std will be 1
                train_examples{i,:} = train_examples{i,:} ./ m.std;
            end
            % end of standardisation process
            
            %% copies train examples as a new train_examples field for the returned m structure
            m.train_examples = train_examples;
            %% copies train examples as a new train_labels field for the returned m structure
            m.train_labels = train_labels;
            %% copies train k (number of nearest neighbours) as a new k field for the returned m structure
            m.k = k;
        
        end
        %% --compute distance between current example to classify and other examples in training data
        %% --For the k training examples with the lowest distance scores, find the k corresponding class labels 
        %% --Compute the mode (most common) class label across these k values to give a prediction 
        function predictions = predict(m, test_examples)
            %% initialise categorical array predictions
            predictions = categorical;
            %% loop through all rows in test_examples table (testing examples)
            for i=1:size(test_examples,1)
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                %% copy current row data from test_examples table in this_test_example 
                this_test_example = test_examples{i,:};
                
                % start of standardisation process (increase model accuracy)
                %% substract mean from current training example to have feature centered at zero
                this_test_example = this_test_example - m.mean;
                %% scales down features with big spreads and scales up features with small spreads, all std will be 1                
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                %% copy prediction returned from predict_one function in this_prediction by passing the model m and the current standardized test data this_test_example
                this_prediction = myknn.predict_one(m, this_test_example);
                %% add prediction to prediction array
                predictions(end+1) = this_prediction;
            
            end
        
		end
        %% takes the struct m (our model) and the current test example (this_test_example)
        %% calculate distances between the current test example and all training examples in m, to find array of first k values near current test example 
        %% return the prediction for proposed test example
        function prediction = predict_one(m, this_test_example)
            %% calculate distances between the current the current standardized test data (this_test_example) and all training examples saved in struct m (model), then copies this in distances array
            distances = myknn.calculate_distances(m, this_test_example);
            %% copy from m.train_examples() array of indicies of first k neighbours -closest to current test example in this_test_example 
            neighbour_indices = myknn.find_nn_indices(m, distances);
            %% copy value of most common categorical class associated with current test example in prediction 
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end
        %% calculate the distance between current the current standardized test data and all training examples stored in struct m (model)
        %%
        %% return distances array
        function distances = calculate_distances(m, this_test_example)
            %% initialise distances array
			distances = [];
            %% loop through all training examples calculate distance and saves distance in distances array
			for i=1:size(m.train_examples,1)
                %% save copy of current training example in this_training_example
				this_training_example = m.train_examples{i,:};
                %% copy calculated distance between current test example and current training example in this_distance
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                %% add distance in distances array
                distances(end+1) = this_distance;
            end
        
		end
        %% --calculate euclidien or straight line distance between two examples
        %% --by treating training example array (p)  and testing example array (q) as single points
        %% return calculated distance
        function distance = calculate_distance(p, q)
            %% apply pithagoras theorem and store final result in distance
            
            %% difference between elements in array p and q store result in differences
			differences = q - p;
            %% square root of differences store resulting array in squares
            squares = differences .^ 2;
            %% sum squares array
            total = sum(squares);
            %% square root of distance
            distance = sqrt(total);
        
		end
        %% takes array of distances (distances) from current test example to all training examples and structure m (model)
        %% return array of original table indices (neighbour_indices) for first k "nearest neighbourghs" to the current test example   
        
        function neighbour_indices = find_nn_indices(m, distances)
            %% sort array of distances and returns sorted array and original indices array
			[sorted, indices] = sort(distances);
            %% copy the first k original indicies nearest to the current test example
            neighbour_indices = indices(1:m.k);
        
        end
        %% takes array of original table indicies (neighbour_indices) and structure m (model)
        %% find most common value in m.training_examples categorical array for first k values, i.e the prediction
        %% the most common value will be the (categorical) class associated with the current example
        %% return prediction
        function prediction = make_prediction(m, neighbour_indices)
            %% copy array of labels in m.train_labels using array of (original) k neighbour indicies from find_nn_indices  
			neighbour_labels = m.train_labels(neighbour_indices)
            %% point out the most common result (most likely class) in neighbour labels array, copy value in predication
            prediction = mode(neighbour_labels)
        
		end

    end
end
