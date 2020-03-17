from gtts import gTTS
from threading import Thread
from datetime import datetime
from time import sleep
import pyglet
import os
IP="192.168.0.16"

class Speech(Thread):
    def __init__(self, text, language_code='en'):
        Thread.__init__(self)
        self.text = text
        self.language_code = language_code
        now = datetime.now()
        self.filename = now.strftime("%d-%m-%Y-%H-%M-%S.mp3")
        self.tts = gTTS(self.text, lang=self.language_code)
        if os.name != "posix":
            self.filename = 'tmp/'+self.filename
        else:
            #server apache2 for google home support
            import socket
            import pychromecast
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.local_ip=s.getsockname()[0]
            s.close()
            print("[*] local ip : "+self.local_ip)
            self.castdevice = pychromecast.Chromecast(IP)
            self.castdevice.wait()
            self.vol_prec=self.castdevice.status.volume_level
            self.castdevice.set_volume(0.0) #set volume 0 for not hear the BEEEP

    def writeToDisk(self):
        if os.name != "posix":
            with open(self.filename, 'wb') as f:
                print("[*] Writing to disk ..")
                self.tts.write_to_fp(f)
                print("[*] Done")
        else:
            try:
                os.mkdir("/var/www/html/mp3_cache/")
            except:
                pass
            if not os.path.isfile("/var/www/html/mp3_cache/"+self.filename):
                print("[*] Creating file for text : "+self.text)
                self.tts.save("/var/www/html/mp3_cache/"+self.filename)
                print("[*] Done")

    def play(self):
        if os.name != "posix":
            music = pyglet.media.load(self.filename, streaming=False)    
            music.play()
            sleep(music.duration) #prevent from killing
        else:
            mc = castdevice.media_controller
            mc.play_media("http://"+local_ip+"/mp3_cache/"+self.filename, "audio/mp3")
            mc.block_until_active()
            mc.pause() #prepare audio and pause...
            time.sleep(0.5)
            self.castdevice.set_volume(self.vol_prec) #setting volume to precedent value
            time.sleep(0.2)
            print("[*] Playing mp3 ...")
            mc.play() #play the mp3
            while not mc.status.player_is_idle:
                time.sleep(0.5)
            mc.stop()
            self.castdevice.quit_app()

    def close(self):
        print("[!] Removing tmp file "+self.filename)
        if os.name != "posix":
            os.remove(self.filename)
            print("[*] Done")
        else:
            os.remove("var/www/html/mp3_cache/"+self.filename)
            print("[*] Done")
    def run(self):
        self.writeToDisk()
        self.play()
        self.close()
