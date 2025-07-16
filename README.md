# Fingers_Counting_CNN
The purpose of this project is to create a system capable of counting how many fingers are raised in a photo - the scenario set for the data acquisition session involves white background and one hand photos.

A dataset was created with the participation of 11 subjects in a laboratory room with a white wall background. Each subject used both hands, one at a time, to record approximately 30-second video sequences for each hand gesture representing the numbers 0 to 5 (six classes in total). For each subject, two videos were recorded per class, from which around 60 images per class were extracted, resulting in a total of 3866 images. During recording, the camera angles and distances were varied to ensure greater diversity. The dataset was split into 80% for training (3092 images), 10% for validation (387 images), and 10% for testing (387 images). 

A CNN model has been trained on 4 versions of the original dataset in order to get the optimal solution considering the time for processing and predicting:
observation: the images were dimensionally reduced from 1080x1920 to 360x640 due to our simple scenario and lack of details in images
- the original dataset - color, not augmented
- the original dataset - color, augmented (+50% training images) using gaussian noise, rotation and contrast augmentation
- the original dataset - grayscale, not augmented
- the original dataset - grayscale, augmented (+50% training images) using gaussian noise, rotation and contrast augmentation

Convolutional neural networks (CNNs) proved highly effective in accurately classifying images based on the number of raised fingers, achieving accuracies of up to 99–100% in controlled and augmented scenarios. Dataset augmentation techniques—such as rotations, Gaussian noise, and contrast adjustments—significantly improved model performance, reduced overfitting, and enhanced generalization. Converting images to grayscale simplified the model’s complexity without negatively impacting accuracy, and when combined with augmentation, led to a 99% accuracy rate.

Merging the custom dataset with a public one (https://www.kaggle.com/datasets/koryakinp/fingers/data) enabled training of a highly robust model, which achieved 100% accuracy on a test set from the same distribution. However, testing the model on a newly collected dataset (219 new color images taken in real life) under real-world conditions resulted in lower performance (~84% accuracy), revealing its sensitivity to unseen variations during training. To improve generalization in real-life contexts, it's necessary to expand the dataset with images from diverse environments, backgrounds, and lighting conditions, as well as optimize the network architecture for greater adaptability.

The experimental results confirm the feasibility of using CNNs for automatic recognition of the number of raised fingers in an image. Careful data processing, augmentation, and testing on varied datasets demonstrated both the model's effectiveness in controlled settings and the ongoing need for dataset expansion to ensure robust real-world generalization. The proposed system offers a solid foundation for future developments, including integration into mobile apps or gesture-based assistive interfaces.


