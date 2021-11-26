# Nudetect: Neural network to detect nudity (fork of NudeNet)

**Default Detector classes:**
| Class name          | Description                                   |
|---------------------|-----------------------------------------------|
| EXPOSED_ANUS        | Exposed Anus; Any gender                      |
| EXPOSED_ARMPITS     | Exposed Armpits; Any gender                   |
| COVERED_BELLY       | Provocative, but covered Belly; Any gender    |
| EXPOSED_BELLY       | Exposed Belly; Any gender                     |
| COVERED_BUTTOCKS    | Provocative, but covered Buttocks; Any gender |
| EXPOSED_BUTTOCKS    | Exposed Buttocks; Any gender                  |
| FACE_F              | Female Face                                   |
| FACE_M              | Male Face                                     |
| COVERED_FEET        | Covered Feet; Any gender                      |
| EXPOSED_FEET        | Exposed Feet; Any gender                      |
| COVERED_BREAST_F    | Provocative, but covered Breast; Female       |
| EXPOSED_BREAST_F    | Exposed Breast; Female                        |
| COVERED_GENITALIA_F | Provocative, but covered Genitalia; Female    |
| EXPOSED_GENITALIA_F | Exposed Genitalia; Female                     |
| EXPOSED_BREAST_M    | Exposed Breast; Male                          |
| EXPOSED_GENITALIA_M | Exposed Genitalia; Male                       |

# As Python module
**Installation**:
```bash
pip install --upgrade nudetect
```

**Detector Usage**:

```python
# Import module
import nudetect

# initialize detector (downloads the checkpoint file automatically the first time)
MODEL_DOWNLOAD_URL = nudetect.MODEL_CHECKPOINT_URL
# ensure that the model is present somewhere

detector = nudetect.Detector('path_to_downloaded_model')

# Detect single image
detector.detect('path_to_image')
# fast mode is ~3x faster compared to default mode with slightly lower accuracy.
detector.detect('path_to_image', mode='fast')
# Returns [{'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...]

```
