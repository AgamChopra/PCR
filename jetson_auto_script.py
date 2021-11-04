import RPi.GPIO as GPIO
import model1 as mod
import jetson.utils as jetutil

R = 40
L = 38

GPIO.setmode(GPIO.BOARD)

GPIO.setup(L, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(R, GPIO.OUT, initial=GPIO.LOW)

with mod.torch.no_grad():
	model = mod.auto().to('cuda')
	model.load_state_dict(mod.torch.load(r'/home/agam-nv/model1.pth'))
	model.eval()
	camera = jetutil.gstCamera(360, 360, "0")

	for _ in range(100):

		img_raw, width, height = camera.CaptureRGBA(zeroCopy=1)
		img = jetutil.cudaAllocMapped(width=width, height=height, format='rgb8')
		jetutil.cudaConvertColor(img_raw, img)
		jetutil.cudaDeviceSynchronize()
		tensor = mod.torch.from_numpy(jetutil.cudaToNumpy(img)).T.unsqueeze(0)
		y = model(tensor.to('cuda', dtype=mod.torch.float)).squeeze()
		GPIO.output(L, GPIO.HIGH) if y[0] > 0.5 else GPIO.output(L, GPIO.LOW)
		GPIO.output(R, GPIO.HIGH) if y[1] > 0.5 else GPIO.output(R, GPIO.LOW)
	
GPIO.cleanup()