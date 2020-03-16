from gtts import gTTS
from threading import Thread
from datetime import datetime
from time import sleep
import pyglet
import os

class Speech(Thread):
    def __init__(self, text, language_code='en'):
        Thread.__init__(self)
        self.text = text
        self.language_code = language_code
        self.tts = gTTS(self.text, lang=self.language_code)
        now = datetime.now()
        self.filename = now.strftime("%d-%m-%Y-%H-%M-%S.mp3")
        self.filename = 'tmp/'+self.filename
    def writeToDisk(self):
        with open(self.filename, 'wb') as f:
            print("[*] Writing to disk ..")
            self.tts.write_to_fp(f)
            print("[*] Done")

    def play(self):
        music = pyglet.media.load(self.filename, streaming=False)    
        music.play()
        sleep(music.duration) #prevent from killing

    def close(self):
        print("[!] Removing tmp file "+self.filename)
        os.remove(self.filename)
        print("[*] Done")
    def run(self):
        self.writeToDisk()
        self.play()
        self.close()
