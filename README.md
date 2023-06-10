# Music Mood Ring

Given a set of lyrics for a song, we are attempting to predict the mood of the song. 
The original paper uses lyrics as well as audio, however, we are simplifying this to be just lyrics. This is a multi-class classification problem using reinforcement learning. 
The applications of this project would be used to do something such as making playlists on spotify (or other music platform) or radio genres. 
Ultimately, it would help users find music in a mood category of their choosing.

Our loss/accuracy:
For classification(model_mood.py): loss: .979; accuracy: .536
For regression(model_val.py): loss: .037; r_squared: .134

See our paper for more information and detailed results! 

Please note: Our larger dataset is too large to upload to git. Please download it yourself and you will be able to run our code. Here is the link:
 https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence?resource=download.