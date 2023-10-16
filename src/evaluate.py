import torch

def evaluate(model, dataloader, loss_function, tokenizer):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    label_to_text = {
        0: "Negative",
        1: "Positive",
        2: "Neutral",
        3: "Irrelevant"
    }

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y.long())
            total_loss += loss.item()

            predicted_classes = torch.argmax(predictions, dim=1)
            
            correct_predictions += (predicted_classes == batch_y.long()).sum().item()

            # Print the sentence, predicted and actual labels
            # for i in range(batch_X.size(0)):
            #     sentence = tokenizer.to_text(batch_X[i])
            #     predicted_label = label_to_text[predicted_classes[i].item()]
            #     actual_label = label_to_text[batch_y[i].item()]
                
            #     print(f"Sentence: {sentence}")
            #     print(f"Predicted: {predicted_label}")
            #     print(f"Actual: {actual_label}\n")

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)

    return average_loss, accuracy