from torchvision import datasets

dataset = datasets.ImageFolder('./test/test_data', transform=None)
gt_labels = {str(s[0].split('/')[-1]): int(s[1]) for s in dataset.samples}
gt_labels_string = [' '.join([str(s) for s in l])
                    for l in list(gt_labels.items())]
with open('./test/test_label', 'w') as file_writer:
    file_writer.write("\n".join(gt_labels_string))
