# Scene Image Classifier

I built this for my deep learning course. It's a convolutional neural network (CNN) that I trained from scratch to look at a photo and say which of six scenes it shows: **buildings, forest, glacier, mountain, sea or street**. I also made a small web app with Streamlit, so anyone can upload a photo and try it.

**Try the app:** <add the link after deploying>

## How well does it work?

On 456 test photos that the model never saw during training, it got **80.3%** right.

- **Easiest scene:** forest. Its green texture is easy to spot.
- **Hardest scene:** sea. The model often mixes it up with glaciers and mountains, which also have lots of blue and white.
- **Other common mistake:** streets and buildings get confused, because photos of one usually contain the other.

I only had about 2,000 training photos and a fairly small network, so I'm happy with 80% for a first version. Training and validation accuracy stayed close together, which tells me the model wasn't just memorising the images.

## How I built it

1. **Data:** 3,000 photos in six folders, split 70% training, 15% validation and 15% test.
2. **Preparing the images:** every photo is resized to 150 x 150 and scaled to values between 0 and 1. I added random rotation, shifting, zooming and flipping to the training photos only, so the model sees more variety. The validation and test photos are left as they are, because they're there to measure the model.
3. **The model:** three blocks of convolution and max pooling, then a dense layer with dropout, then six outputs with softmax.
4. **Training:** 20 epochs with the Adam optimizer. It took about 25 seconds per epoch on my laptop's CPU.
5. **Checking the results:** test accuracy, a confusion matrix and a classification report.
6. **The app:** a Streamlit page that prepares an uploaded photo the same way as the training photos and shows the prediction and a score for each scene.

## Things I learned

- My first train/validation/test split went wrong. I re-ran it on top of an old folder, and some photos ended up in two sets. I only noticed because two sets that should have been the same size weren't. Now the notebook deletes the old folder before splitting.
- A new photo has to be prepared exactly like the training photos. If the size or scaling is different, the model still gives an answer, just a wrong one, and nothing warns you.
- Looking at the confusion matrix taught me more than the accuracy number did.

## What's in this repository

- `app.py` is the Streamlit app
- `scene_cnn_model_small.keras` is the trained model
- `Cnn 6 class_classifier.ipynb` is the full notebook, from splitting the data to evaluating the model
- `requirements.txt` lists the libraries the app needs

## Running it on your computer

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Dataset

Intel Image Dataset from Kaggle (a 3,000-image version with six classes). I haven't included it here because of its size.

## What I'd try next

Training for longer with early stopping, and using a pretrained model (transfer learning) to see how much higher the accuracy can go.

*Made by <Piyush Tak>*
