% Convert class labels from scalars to one-hot vectors
function labels_onehot = onehot_labels(labels, num_classes)
    labels = labels(:);
    labels_onehot = full(sparse(1:length(labels),labels,1));
    % num_labels = size(labels,1);
    % labels_onehot = zeros(num_labels, num_classes);
    % for i = 1:num_labels
    %     labels_onehot(i, labels(i)) = 1;
    % end
end