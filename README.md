# VideoMasker

**Secure and convenient Client-Side Video Automatic De-Identification Program** <br/>
1. Open the video
2. Press the 'Analyze' button
3. Check the thumbnail on the right and select what to exclude from de-identification (optional)
4. Press the 'De-Id' button
5. It's really easy, right?

Videos will __not be uploaded to the server__.  <br/>
It is only processed on __your device__.

## Requirements

Developed in `Python 3.11`
```text
pyside6
pytorch
torchvision
transformers
scikit-learn
opencv-python
huggingface-hub
imgbeddings
ultralytics
scipy
supervision
```

## Getting Started

```bash
git clone https://github.com/dh031200/VideoMasker
pip install -r requirements.txt
python main.py
```
## Screenshots/Demo

### Selectively de-identify targets

![Selectively](https://github.com/dh031200/VideoMasker/blob/main/assets/selectively.png)
### Blur

![Blur](https://github.com/dh031200/VideoMasker/blob/main/assets/blur.png)
### Pixelate

![Pixelate](https://github.com/dh031200/VideoMasker/blob/main/assets/pixelate.png)
### Emoji

![Emoji](https://github.com/dh031200/VideoMasker/blob/main/assets/emoji.png)

## Contributing

Feel free to fork the repository, make changes, and create pull requests. Any contributions to improve the project are welcome!

## License

[AGPL-3.0 license](https://github.com/dh031200/VideoMasker/blob/main/LICENSE)

## Contact Information

If you have any questions, please contact us.
<br/>[[techniflows - Contact]](https://techniflows.com/en/contact/)

## Acknowledgements
Sample Video(Happy Monica Geller Scenes | Logoless 1080p) : [https://www.youtube.com/watch?v=8FZ2lDSYgvs](https://www.youtube.com/watch?v=8FZ2lDSYgvs) <br/>
Ultralytics-YOLOv8 : [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)<br/>
Roboflow-supervision : [https://github.com/roboflow/supervision](https://github.com/roboflow/supervision)<br/>
ifzhang-ByteTrack : [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)<br/>
