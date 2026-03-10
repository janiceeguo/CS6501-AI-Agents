# Vision-Language Models (VLM):

- Jupyter notebook (all code) in `vlm.ipynb`
- Separate code and models for each task in each `.py` file
- Video and images in the `inputs` folder
- Separate outputs for each exercise in the `outputs` folder

## Exercise 1

I initially had a bit of trouble getting llava to run reliably as it would either hang or crash, but was able to set up the full pipeline using a smaller version, combined with downsizing the image. However, even with all these adjustments, each response from the model still took around 4-5 minutes. The final conversation documented in `output2.txt` shows that the model is able to pick out key details from a simple image, and able to make recomendations based on previous queries.

## Exercise 2

I used a video that captures a person from the knees upwards. The first person enters at 00:02 and exits at 00:04. The next person enters at 00:52 and exits at 00:57. A third person follows closely, entering at 00:56 and exiting at 00:57. The last person enters at 01:13 and exits at 01:17. From the output, we see that llava hallucinates individuals and is unable to correctly identify entry and exit points. Looking at the jupyter notebook we can also see that the model disagrees with itself on different runs for the same video input. Several things can explain the poor performance:
1. The llava model is small. A bigger model would take more time, but would possibly be more accurate in its identification of whether a person is in the frame or not, and whether they are entering or exiting.
2. The frames were taken 2 seconds apart for the model to analyze. Some individuals walked very fast, completing an entry and an exit in less than 2 seconds. This gap could cause the model to miss certain individuals.
3. Some individuals overlapped. Around 1 minute, two individuals appear in close succession. The model is only prompted to say whether there is a person in the frame or not, and does not differentiate between how many individuals are in the frame. This can confuse the model and cause it to incorrectly label exits and entries. 
4. The video was filmed manually, which includes minor motion and blur. It was also filmed at a 0.5 camera lense, which could have caused some distortion that confused the model. Objects such as wall outlets could also be confusing.