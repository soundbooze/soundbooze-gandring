import gtk
import glib
import cairo

import cv2

import os
import sys
import jack
import vamp
import numpy
import psutil
import signal
import librosa
import threading

import PIL.Image as Image
from skimage.measure import compare_ssim

GANDRING    = "GandRing"
DESC        = "Guitar And Ring Buffer"

WIDTH       = 1920
HEIGHT      = 1080
BORDER      = 22

ICON        = "/usr/share/pixmaps/jack-keyboard.png"
FACECASCADE = "haarcascades/haarcascade_frontalface_default.xml"

guitar      = None
system      = None

stopThreads = False

rg, eg, dg  = 0, 0, 0
rs, es, ds  = 0, 0, 0

currentTempo, tempoGram, predictTempo = 0, None, 0
chords      = ""

# --- JACK Thread ---

signal.signal(signal.SIGINT, signal.SIG_DFL)
client = jack.Client(GANDRING)

def db(rms):
    db = 20.0 * (numpy.log(rms) / numpy.log(10))
    return db
  
def rms(y):
    rms = 0;
    for i in y:
      rms += i * i #numpy.power(i, 2);
    rms = rms / len(y);
    rms = numpy.sqrt(rms);
    return rms
  
def energy(y):
    energy = 0;
    for i in y:
      energy += i * i #numpy.power(i, 2);
    return energy

@client.set_process_callback
def process(frames):

    global guitar, system
    global rg, eg, dg
    global rs, es, ds

    guitar = client.inports[0].get_array()
    system = client.inports[1].get_array()

    rg = rms(guitar)
    eg = energy(guitar)
    dg = db(rg)

    rs = rms(system)
    es = energy(system)
    ds = db(rs)

    #zero_crossings = librosa.zero_crossings(guitar[0:frames], pad=False)
    #spectral_rolloff = librosa.feature.spectral_rolloff(guitar, sr=client.samplerate)[0]
    #spectral_centroids = librosa.feature.spectral_centroid(guitar, sr=client.samplerate)[0]
    #X = librosa.stft(guitar)
    #Xdb = librosa.amplitude_to_db(abs(X))
    #print Xdb, spectral_rolloff, spectral_centroids, numpy.sum(zero_crossings)

@client.set_shutdown_callback
def shutdown(status, reason):
    print("JACK shutdown!")
    print("status:", status)
    print("reason:", reason)
    event.set()

event = threading.Event()

client.inports.register("guitar")
client.inports.register("system")

def JackThread():

    with client:

        try:
            event.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user")

j = threading.Thread(target=JackThread)
j.start()

# --- Feature Thread ---

class RingBuffer:

    def __init__(self, size):
        self.data = [0 for i in xrange(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

def FeatureThread():

    global currentTempo, tempoGram, predictTempo
    global chords

    rb = RingBuffer(8)

    while True:

        os.system('jack_capture_ms -d 1000 out.wav > /dev/null 2>&1')
        y, sr = librosa.load('out.wav', sr=client.samplerate)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        rb.append(int(numpy.around(tempo)))
        tempogram = numpy.array(rb.get())
        unique, counts = numpy.unique(tempogram, return_counts=True)
        max_unique_counts = numpy.asarray((unique, counts)).T
        currentTempo, tempoGram, predictTempo = tempo[0], rb.get(), max_unique_counts[0][0]

        vdata = vamp.collect(y, client.samplerate, "nnls-chroma:chordino")

        idx = 0
        for i in vdata['list']:
            c = vdata['list'][idx]['label']
            if c != 'N':
                chords = c
            idx = idx + 1

        os.system('rm -f out.wav')

        if stopThreads:
            break

f = threading.Thread(target=FeatureThread)
f.start()

# --- GandRing Thread ---

class GandRing(gtk.Window):
   
    def __init__(self):

        super(GandRing, self).__init__()

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(FACECASCADE)

        self.set_title("G And Ring")
        self.connect("destroy", self.on_destroy)
        self.set_position(gtk.WIN_POS_CENTER)
        self.maximize()
                
        self.darea = gtk.DrawingArea()
        self.darea.connect("expose-event", self.expose)
                
        self.add(self.darea)
        self.show_all()

        self.timer = True
        self.interval = 50
        glib.timeout_add(self.interval, self.on_timer)
     
        self.prevframe = None

        self.egmin, self.esmin = 0,0
        self.egmax, self.esmax = numpy.iinfo(numpy.int16).min, numpy.iinfo(numpy.int16).min

    def on_destroy(self, widget):
        self.timer = False
        gtk.main_quit()
        #self.cap.release()
        stopThreads = True

    def on_timer(self):
        if not self.timer: 
            return False
        self.darea.queue_draw()
        return True

    def _similar(self, prev, curr):
        a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        sim, _ = compare_ssim(numpy.array(a), numpy.array(b), full=True)
        return sim

    def _clear(self, cr):
        cr.set_source_rgb(0.08, 0.08, 0.08)
        cr.rectangle(0, 0, WIDTH, HEIGHT - BORDER)
        cr.fill()

    def _header(self, cr):
        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.rectangle(0, 0, WIDTH, 58)
        cr.fill()

        cr.select_font_face("Courier", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(70)
        cr.set_source_rgb(0.19, 0.19, 0.19)
        cr.move_to(40, 70)
        cr.show_text(GANDRING)
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.move_to(45, 75)
        cr.show_text(GANDRING)

        cr.set_line_width(4)
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.move_to(60, 90)
        cr.line_to(330, 90)
        cr.stroke()

        cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(30)
        cr.set_source_rgb(0.29, 0.29, 0.29)
        cr.move_to(1480, 40)
        cr.show_text(DESC)

        cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(22)
        cr.set_source_rgb(0.29, 0.29, 0.29)
        cr.move_to(1680, 70)
        cr.show_text(jack.version_string())

        ims = cairo.ImageSurface.create_from_png(ICON)
        cr.set_source_surface(ims, WIDTH - 68, 20)
        cr.paint()

    def _crossline(self, cr):
        cr.set_line_width(0.8)
        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.move_to(0, HEIGHT/2)
        cr.line_to(WIDTH, HEIGHT/2)
        cr.move_to(WIDTH/2, 0)
        cr.line_to(WIDTH/2, HEIGHT)
        cr.stroke()

    def _wave(self, cr):
        global guitar, system

        if numpy.sum(guitar) and numpy.sum(system):
            i = 0
            for g, s in zip(guitar, system):
                cr.set_source_rgb(1.0, 0.0, 0.0)
                cr.rectangle(70 + i, 140 + (g*11), 2, 2)
                cr.fill()
                cr.set_source_rgb(1.0, 1.0, 1.0)
                cr.rectangle(70 + i, 180 + (s*11), 2, 2)
                cr.fill()
                i += 1

    def _feature(self, cr):
        global currentTempo, tempoGram, predictTempo
        global chords

        cr.select_font_face("Courier", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(18)
        cr.set_source_rgb(0.39, 0.39, 0.39)
        cr.move_to(70, 230)
        tg = "T: [ "
        for t in tempoGram:
            tg += str(t) + " "
        tg += "]"
        cr.show_text(tg)
        cr.move_to(70, 250)
        cr.show_text("C: " + "% 0.2f" % currentTempo + " P: " + "% 0.2f" % predictTempo + " BPM")

        cr.move_to(70, 290)
        cr.set_font_size(32)
        cr.show_text(chords)

    def _from_pil(self, im, alpha=1.0, format=cairo.FORMAT_ARGB32):
        assert format in (cairo.FORMAT_RGB24, cairo.FORMAT_ARGB32), "Unsupported pixel format: %s" % format
        if 'A' not in im.getbands():
            im.putalpha(int(alpha * 256.))
        arr = bytearray(im.tobytes('raw', 'BGRa'))
        surface = cairo.ImageSurface.create_for_data(arr, format, im.width, im.height)
        return surface
        
    def _camera(self, cr):
        ret, frame = self.cap.read()
        h, w, _ = frame.shape

        cr.set_source_rgb(0.2, 0.2, 0.2)
        cr.rectangle(WIDTH/2 - (w+16)/2, HEIGHT/2 - (h+16)/2, w + 16, h + 16)
        cr.fill()

        _, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
        im = Image.fromarray(thresh)
        ims = self._from_pil(im, format=cairo.FORMAT_RGB24)
        cr.set_source_surface(ims, WIDTH/2 - w/2, HEIGHT/2 - h/2)

        cr.paint()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (fx,fy,fw,fh) in faces:
            cr.set_font_size(22)
            cr.set_source_rgb(0.39, 0.39, 0.39)
            cr.move_to(640, 280)
            cr.show_text("(" + "% d" % fx + "% d" % fy + ")")

            cr.set_source_rgb(1.0, 0.0, 0.0)
            cr.set_line_width(3.0)
            cr.rectangle((WIDTH/2 - (w+16)/2)+fx, (HEIGHT/2 - (h+16)/2)+fy, fw, fh)
            cr.stroke()

        if self.prevframe is not None:
            ssim = self._similar(self.prevframe, thresh)
            cr.set_font_size(22)
            cr.set_source_rgb(0.39, 0.39, 0.39)
            cr.move_to(1100, 820)
            cr.show_text("% 0.9f" % ssim)

        self.prevframe = thresh

    def _footer(self, cr):
        global rg, eg, dg
        global rs, es, ds

        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.rectangle(0, HEIGHT - 40, WIDTH, 40)
        cr.fill()

        cr.select_font_face("Courier", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(28)
        cr.set_source_rgb(0.29, 0.29, 0.29)
        cr.move_to(WIDTH - 430, HEIGHT - 30)

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()[2]

        cr.show_text("CPU: " + "% 0.2f" % cpu + " Mem: " + "% 0.2f" % mem)

        cr.rectangle(40, 700, 60, 300)
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.set_line_width(2)
        cr.stroke()

        cr.rectangle(110, 700, 60, 300)
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.set_line_width(2)
        cr.stroke()
        
        jc = client.cpu_load()
        rt = client.realtime
        sr = client.samplerate
        bs = client.blocksize

        cr.move_to(40, HEIGHT - 30)
        cr.show_text("Load: " + "% 0.2f" % jc + " RT: " + "% d" % rt + " SR: " + "% d" % sr + " BS: " + "% d" % bs)

        cr.set_font_size(22)
        cr.move_to(180, 950)
        cr.show_text("% 0.2f" % rg + " (" + "% 0.2f" % dg + " dB)")
        cr.move_to(180, 980)
        cr.show_text("% 0.2f" % rs + " (" + "% 0.2f" % ds + " dB)")

        if eg > self.egmax:
            self.egmax = eg
        if es > self.esmax:
            self.esmax = es
        if eg < self.egmin:
            self.egmin = eg
        if es < self.egmin:
            self.egmin = es

        eg = 100 * ((eg - self.egmin) / (self.egmax - self.egmin))
        es = 100 * ((es - self.esmin) / (self.esmax - self.esmin))

        l = (100 - eg) * 3
        cr.rectangle(40, 700 + l, 60, 300 - l)
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.fill()

        l = (100 - es) * 3
        cr.rectangle(110, 700 + l, 60, 300 - l)
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.fill()

    def expose(self, widget, event):

        cr = widget.window.cairo_create()

        self._clear(cr)
        self._header(cr)
        self._crossline(cr)
        self._wave(cr)
        self._feature(cr)
        self._camera(cr)
        self._footer(cr)

if __name__ == '__main__':

    GandRing()
    gtk.main() 
    cv2.destroyAllWindows()
