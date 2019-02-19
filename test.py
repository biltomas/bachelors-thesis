def test(tstLabels, tstData, model):
    tstLabels_split = np.split(tstLabels, 1)
    tstData_split = np.split(tstData, 1)

    # print((tstLabels_split[0]))

    for i in range(len(tstData_split)):
        temp_acc = 0.0
        batch_data = torch.from_numpy(tstData_split[i])
        batch_labels = torch.from_numpy(tstLabels_split[i])
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        outputs = model(batch_data)
        # print(outputs)
        max_scores, pred_labels = torch.max(outputs, 1)
        # print(max_scores)
        # print(pred_labels)

        accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size*2)
        temp_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size*2)
        # print ('Test batch Accuracy: {:.4f}' .format(temp_acc))
        # total_acc += accuracy_acc

    accuracy_acc = accuracy_acc/len(tstData_split)
    print ('Test Accuracy: {:.4f}' .format(accuracy_acc))
    return accuracy_acc