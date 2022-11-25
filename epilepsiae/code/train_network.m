function [net, out] = train_network(name, file, varargin)
    %TRAIN_NETWORK Summary of this function goes here
    %   Detailed explanation goes here
        input_data = varargin{1};
        output_data = varargin{2};
        validation_data = varargin{3};
        result = [];
        if strcmp(name,"CNN")
            % example: train_network("cnn", imageTrain, trgImageTrain, 
            % {imageTest, trgImageTest}, [])
            weights = varargin{4}; % can also be [] for no use
            net = conv_nn(input_data, output_data, validation_data, weights);
            result = classify(net, validation_data{1});
        elseif strcmp(name, "LSTM")
            % example: net = 
            % train_network("lstm", dataTrain', trgTrain', {dataTest', trgTest}, 
            %   20, [5,5,1,1]);
            numHiddenUnits = varargin{4};
            weights = varargin{5}; % can also be [] for no use
            net = lstm_network(input_data, output_data, numHiddenUnits,29, weights);
            result = classify(net, validation_data{1});
        elseif strcmp(name, "SNN")
            % example: train_network("shallow", dtr', ...
            % trgTrain', {dataTest, trgTest},50, 5, [1 5 1 1]);
            noHiddenLayers = varargin{4};
            weights = varargin{5};
            net = shallow_network(input_data, output_data, 20, noHiddenLayers, weights);
            result = net(validation_data{1});
        elseif strcmp(name, "RNN")
            weights = varargin{5};
            net = train_rec(input_data, output_data, varargin{4}, 5, 29, 5,"traingdx", weights);
            result = net(validation_data{1});
        end

        save("../data/"+file+".m", "net");
        size(result);
        size(validation_data{2});
        [sens2, spec2] = report_spec_sens(validation_data{2}, result, 2);
        [sens3, spec3] = report_spec_sens(validation_data{2}, result, 3);
        figure
        plotconfusion(validation_data{2}, result);
        out = sprintf("Prediction: (%f, %f); Detection: (%f, %f)\n", sens2, spec2, sens3, spec3);
    end