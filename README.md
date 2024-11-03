# DataScience-Models-and-Deployment : Chapter-4 Speech-to-Text for converting speech into text

## 1) Installig Python SpeechRecognition [ref](https://pypi.org/project/SpeechRecognition/)
```shell
pip install SpeechRecognition
```

## 2) Import Library
```py
import speech_recognition as sr
```

## 3) Listen to the Audio File
```py
import IPython.display as ipd
ipd.Audio('/content/sample_data/harvard.wav')
```

## 4) Plotting a Waveform
```py
import matplotlib.pyplot as plt
import librosa.display

x, sr = librosa.load('/content/sample_data/harvard.wav')
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()
```

![01](/01.png)

## 5) Convert Voice Recordings to Text

```py
import speech_recognition as sr

s2t = sr.Recognizer()
text = ""

with sr.AudioFile('/content/sample_data/harvard.wav') as source:
    audio_length = int(source.DURATION)  # Get the duration of the audio in seconds
    for start in range(0, audio_length, 30):  # Process in 30-second chunks
        source_offset = start
        source_duration = min(30, audio_length - start)
        
        sound2text = s2t.record(source, offset=source_offset, duration=source_duration)
        
        try:
            text += s2t.recognize_google(sound2text) + " "
        except sr.UnknownValueError:
            print("Could not understand audio in this section")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")

print(text)
```

```shell
the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun
```

---

[Goto main](https://github.com/KOPE-Solution/DataScience-Models-and-Deployment/tree/main)
