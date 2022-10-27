function net = shallow_network(P, T, sizeHidden, noHiddenLayers)
    sz_P = size(P);

    net = feedforwardnet(repmat(sizeHidden, 1, noHiddenLayers), "traingd");
    net = configure(net, P, T);
    net.layers{3}.transferFcn = "softmax";
    view(net);
    net = train(net, P, T);
end