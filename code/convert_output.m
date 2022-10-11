function out = convert_output(in)
    % expects in with size 10 x Noutputs
    % formats the output: yields a one-hot encoded vector for each result
    s = size(in);
    out = zeros(s);
    for i = 1:s(2)
        el = in(:,i);
        ind = find(el==max(el));
        out(ind(1),i)=1;
    end
end



% old one

% function out = convert_output(network_output)
%     % formats the output: yields a one-hot encoded vector for each result
%     out = [];
%     s = size(network_output);
%     for i = 1:s(1)
%         el = network_output(i,:);
%         aid = zeros(1, s(2));
%         [~, idx] = max(el); % determines the max element
%         aid(idx) = 1; % one-hot encoding
%         out = [out; aid];
%     end
% end