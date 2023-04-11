# Create function to convert audio file to wav
def convert_to_wav(filename):
  """Takes an audio file of non .wav format and converts to .wav"""
  # Import audio file
  audio = AudioSegment.from_file(filename)
  
  # Create new filename
  new_filename = filename.split(".")[0] + ".wav"
  
  # Export file as .wav
  audio.export(new_filename, format='wav')
  print(f"Converting {filename} to {new_filename}...")
 
# Test the function
convert_to_wav("call_1.mp3")  #takes "call_1.mp3" not 'call_1.mp3'
# The first function down! Beautiful. Now to convert any audio file to .wav format, you can pass the filename to convert_to_wav(). Creating functions like this at the start of your projects saves plenty of coding later on.
---------

def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)
  
  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment

# Try the function
call_1_audio_segment = show_pydub_stats("call_1.wav")
# output:
#     Channels: 2
#     Sample width: 2
#     Frame rate (sample rate): 32000
#     Frame width: 4
#     Length (ms): 54888
# Now you'll be able to find the PyDub attribute parameters of any audio file in one line! It seems call_1.wav has two channels, potentially they could be split using PyDubs's split_to_mono() and transcribed separately.
------------

def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
print(transcribe_audio("call_1.wav"))
# output:
#     hello welcome to Acme studio support line my name is Daniel how can I best help you hey Daniel this is John I've recently bought a smart from you guys 3 weeks ago and I'm already having issues with it I know that's not good to hear John let's let's get your cell number and then we can we can set up a way to fix it for you one number for 17 varies how long do you reckon this is going to try our best to get the steel number will start up this support case I'm just really really really really I've been trying to contact past three 4 days now and I've been put on hold more than an hour and a half so I'm not really happy I kind of wanna get this issue 6 is f***** possible

# You'll notice the recognizer didn't transcribe the words 'fast as' adequately on the last line, starring them out as a potential expletive, this is a reminder speech recognition still isn't perfect. But now you've now got a function which can transcribe the audio of a .wav file with one line of code. They're a bit of effort to setup but once you've got them, helper functions like transcribe_audio() save time and prevent errors later on.
----------


