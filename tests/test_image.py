"""Byte parity of the C image pipeline (clef_image.c, via clef-tool image) with the reference:
PIL decoding (convert("RGB")), Qwen2VLImageProcessor.smart_resize, torchvision's uint8
antialiased bicubic resize, the f32 normalization and patch layout of the processor's
pixel_values, and the position-table interpolation indices and weights.

Usage: .venv/bin/python -B tests/test_image.py [--seed N] [--cases N]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.vision_utils import get_vision_bilinear_indices_and_weights

ROOT = Path(__file__).resolve().parent.parent
MIN_PIXELS, MAX_PIXELS = 65536, 16777216

# A 64x48 baseline JPEG from libjpeg-turbo's `cjpeg -quality 90 -sample 1x1,2x2,1x1`: luma sampled
# below chroma, which libjpeg and PIL decode. The vendored decoder sizes the luma plane from its
# own factors and read it at image resolution (heap over-read, found by review on 2026-10-07); it
# now refuses the layout.
LUMA_UNDER_CHROMA_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMU"
    "FRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
    "FBQUFBQUFBT/wAARCAAwAEADAREAAiIBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm"
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCHVLu0nvoTcWKwXoSK5juJ0YmdG2o4+QLjByAMgARhlUBstEK1LLMPX+qVKlGnNt0qqbTX"
    "LztU7qqpVKcH7SpUiuZKTnUvXly1ZVisJPH4Orm1GjO1XWpKcLTglGU5OOvI1KNNQnC8VFxpycpOalP0IcS4XD5nDK8U2o+1qK3O"
    "qcYSg5qPs/aVHGnUcOaMuacJ2mlJzVoH01TCJRxkcbBVFWbjCb5XCnGUVRjFSUpOKipqnKEZT0ScedzsuSjhJunUTk3B3i1bms23"
    "yyd07XX27+7aKlLlSAfZtTLappk9vbosHl7BcRNIGIAWOTaoZiMqvG4Y+ig/C1Vgcqw9TNcgrShGcqcfZ01NRhyVXCPu1IKatUXL"
    "aV1JJxjJ2SdYdUsXmFBYqjOP1f3FU5pKzSjG16lVqKTcHGDjUjpNqDlKU36eKpPC4vF4vBKToJwqpxsqk5KS5KXL8PNL3nS15akr"
    "SUPawvHwcPUqQo/WMLCly+7UlKahH45YiEvaSVRKddShKcfac7pxnLnmuWLl6GEwrnTj7a8puU+ZO6e/KtJNpaJyUuRPRwXxWJrK"
    "eUNbGJI2t1RWkhuZkEaAsgJcSF/3hEhYHI2hfvHapr6HA160HTlHnjDkqUqjnTUW041IRfNJxlP34y9pyyi3FUqaScWp8eMx+Kx+"
    "AgsnnVjOLpSfK/em7Jy+K0pKm6ekmqcrOnCcFShVadanFwjQhRt7SMnNKraDlGzv7ZVHOlyvmhGnUU/fc4zUqjcmqmDzCjiMV7FO"
    "MacZONSFTllCMY03KN480+enUcnyySpOFWrBNyfu+TNfUJKdCrztNO1nFw0va8VGEk7PmV7JrRrli5XBq1te6M+mSgXE0RMMVrsQ"
    "iFGmQBAu87zjHoQR1BBNceMy114KjC01XjCcqftU1GopxgvaWdOcUoR9+e0uf958EqkvOp5pmVapXpUqztTcpVeWUUpw5VBRtS5f"
    "cnKM4wprllKLo3UPZp0+yq6VHCYbG4XDReKpRjzUqnPBzdKcqjqKkpqzptShJqopxcKcJrmiuTnrRoUMXi6mNnKVOvH3XyuXtKic"
    "1alF8yf7lUuTSdNc1Llk5KnzdscJXo8uIw6mpK8veTV7J31STSsmny2k72lZXZjwTTCxt9N8u8tr64iInkYB5Ddh2IhMqO4QgIxI"
    "boduCNpavpMyxWFynF+0nKaq1Ju/vym5ynTUuVSm/ZJNLXmjyxado1Kahy/N4bJ8BTq05+xU6Tf7yMaLg3UqNwSpRioWlTsm48je"
    "G5rqUWk6nbXcFl9BQw/I6KcVTalK3OpUWuWouRxlKS1jZU4R9km41JyWM8Q54Cln2Axag6SlNvldN04ql7KMf3kpqcFUTnGU7S/5"
    "eRi+dIynh4yg1HWO3utOP2Yr3bc1k4rWEX7qlvFu8+rG9m8UWssVjb6diOaWKdWxak7cFlXcVwx8v5y33WAxnIOuV5pXUFlOfzp0"
    "7c1nye+ouDhBqMKSjOPsrRh+8jq7uMVClKHpwwVKsozrU40pwk3TvKc6kZKEZ1JTi2lNpydSSUXdQtGnLljKPHhMzxOBweEePqzn"
    "Ou3FOUqkZSnGrCEJxlySmk6LXNFRnUhTc5xg+XlIz2VDHZlRpVPaKdeMYqMJKm0rqip88IySceaVre5KNBRbvU9i2sv92WEnOLi7"
    "KUYxjttaHKm1yqMnLRq7+HmnK8T6bFqV6shkaJDKjNA0+bhUYIcDEakMA2V3AcDbjaMmsplhsRShUxWJcaqVObnKpKlNwScJ+9GC"
    "leS50+WUHKcsQ26MZwlHbNaMqlCpDD3VRqpCo46xcJ/vJzahLmldOFOLhThGEoSiopQdMyxmaYudZVaNH91SoyjzqhCNSPNDnlRj"
    "HnhOatUpQTjGMK03NyatdeTzwzOrOeGownS5pxcIXcFGzUV7OP72CjeCadvdapwVNcs36FKrN0qlX2cZPRO/wuKbdoJW93e7aUVa"
    "Tj7vLbUuYhaQ/YBbQJc2kaCKVrZEa1bJMUbShAPlRN652cMA24x7q2wWNxdR1swws5VZxhFtRcY1KdJL2ikm1GrWXPKSnGnCFKoq"
    "cvelZQZHAYytmFXB4ilKrKUGocsrvnqS51VqL2/PKzjU9m4WhOClO8YRckSxdWWMjhsJjOaqoLWNS6S5HBzi1KM1NxT2te05ubhz"
    "VYRnGOpVsPzTrSqxqJ0qftE41J3dOov3kIc1JVOW8oK1SfPSaglUahyYLSX1WjC0J25rLnTu1N6yajdOL0f2lKb92yIY7cXkMemQ"
    "28Vvql5E+wl1Dq6AArEEjwSnJ3naWGMcvz68c6xuEq1KlbERc5T5PeV5Rm4qNKco8nvRlCUXz1KlST5JTpVKtONNQ6cBmtLGzq4J"
    "1aVGpOm17zl7X2SleVSM/c9lL/l7s4wULyl7GPtX8zClSxNfD4hY1QUZc1GdNxjUtKUacHKi7VIt3jL3oTi4ud5NckZfQ1K1fLsX"
    "SeNp+1oyvKh7s41HVg+WMFOLkpRqJRUpq8KrqU7xi5ucuadWnGlUqz52kpWslFtxahFNxn7sdXpzOy5nKS5Ilm6nN9eWjQ+dfpIN"
    "r20yjKiMOcsGVdrgNHkfMT5rgnsPlsPmmHzRwweOnS9jVfJGnL2qcYxqJOUHSk4PljOrJQlU5m4tXlSlJ0/rsRXw08BKliFBYGbr"
    "Sbnac/axduVKE1HmkpShKCUpOcVT5op3p8+DxMcPB47BYn2klzJySVOTk+SMpqnJSUfedJQrcvs1aEKcFZylz5bmH1upSwHM4tXk"
    "pwi5QhV5pcsnNTi96cqcfcpygrU+ePLNUu+tVVNValWacoJtylraS5FJRUG1Hl5PdcpRabUpJtuTS7uZraCS0vbs3bQ2YW4ndUMj"
    "xfLKd+SxQNGn38DIkAG0nj52rRxFH27wrjTbftI0b8kIqUHJv/n5CfNJxUXOjfkjKpOD5D6CrKlh8zeKhOEpt4dKnaq3FtSi2pwa"
    "hJ3lyuq4yjLmqLlbnJ1sMf7OE6ePwVuepGlCV26V1VlLkm4OdW6i3JxXJzcyjLmqNSjDGnJxrzxMcJTqRcHKop1IVJwcIUp0k51V"
    "CLU/Z1IymrwcXOVOc1CV+SOEq/WOelQc5zu3G6to4pN7xlFa9IuTUeZO0iKDUZpbeRP3yi7Ys5aKMfLvLEQ7SQCCQdoH90KV+YH2"
    "pVMPKVCl7W9SOqi6cpqUaMpSjyRTXLFzqezk5ONSbinyqNc8zCUfr2Nq4R3d1zKM1U/iexv7Rya5YTipPlvHR1pSjzSpKJ1VcLg8"
    "bjI/2lBzoqom4SSU+WpJpKqp8rhJxowcnCUZOnSlConuuOWW4DAzw8K1ajRpulT5JTlD2sfZuVWMoN8slJz0cOeM4tc0VOmpKVUa"
    "PsZxoqfs563baV23dXu0rr4rWldJrmTdj//Z"
)


# Progressive 4:2:0 JPEGs with one DC scan per component (cjpeg -scans: "0: 0 0 0 0; 1: 0 0 0 0;
# 2: 0 0 0 0;" then each component's AC), which libjpeg decodes. A non-interleaved luma DC scan
# was walked as MCUs: the 37x21 file was refused and the 32x32 one decoded wrong (review #3).
SEPARATE_DC_SCANS_37X21 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0d"
    "Hx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh7/wgARCAAVACUDASIAAhEBAxEB/8QAGAABAQEBAQAAAAAAAAAAAAAABgQHBQj/2gAIAQEAAAAA8+MmcURrbl+GUc/b"
    "S4JF/8QAFgEBAQEAAAAAAAAAAAAAAAAABwgG/9oACAECEAAAAGyesLTZEPf/xAAXAQEAAwAAAAAAAAAAAAAAAAAFAgQG/9oACAED"
    "EAAAALC+QTiV/8QAKxAAAQMDAwMDAwUAAAAAAAAAAQIDEQQFBgASIQcTMSNhgRQVoTNCUZGi/9oACAEBAAE/ALPRd6ONWfHO9t9O"
    "fjVnwfvbfR/GuoHT8vuUNtSwqAC+sbBBmUp588Qr+x8UHR7uMz9L/nWF0Pe2cTrC8c72z0/xq73yyYjUItwonLndSgLNM0oJS0CR"
    "HcXztJSSQACeBMAgm7XjN80yGqrPuSqClcfCmWrcg04CUp2JO79QyBJClESfAgAUnT6wW1pLF0uNsoXlIC0t1NShtRSSRICiDEg8"
    "+x10xp0OdqfbV+vD2JYU3X21ps1lTUJpWnFiQyVJUrft8KICDAPEkEyBBx+00lJjdTUoCt4bCElJ2lJUQkKB9pn41kmQXG35NU4r"
    "Z1N29lhDaXKltMvub2gowo8IELHgbgUyFCY1iGC2dVsBKf4/br//xAAnEQABAgQFAwUAAAAAAAAAAAABAwQAAgURBhIhMVETFEEi"
    "YXKx0v/aAAgBAgEBPwDG9W7bNrDl89dKzL3EkgFwT5vwN/fwOIcVJcqzZXBI+B/UY4lCtQlTn1BO0LALuF+rrYE68jLb7MP6g57m"
    "f1neP//EACYRAAIBAgQFBQAAAAAAAAAAAAECAwAEBQYRIRIUMUFRIiNzkdL/2gAIAQMBAT8Asbzk9KzPj2IpgyojCNZiACx0JUbk"
    "gDc9h2BB61FPcFAVuiR8Z/VWUKT4pDFINVLbjzWcHN3mdIZ/UqQ8QB6Bi+mv15qTEbriPuGv/9k="
)
SEPARATE_DC_SCANS_32X32 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0d"
    "Hx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh7/wgARCAAgACADASIAAhEBAxEB/8QAGQABAQADAQAAAAAAAAAAAAAABgcDBAUI/9oACAEBAAAAAPPjJnpDbcvhmK1n"
    "53315/gf/8QAFQEBAQAAAAAAAAAAAAAAAAAABwj/2gAIAQIQAAAAbJ6WyX//xAAVAQEBAAAAAAAAAAAAAAAAAAAFBP/aAAgBAxAA"
    "AAChcov/xAAwEAABAwIFAgMGBwAAAAAAAAABAgMRBAUGEhMhMQAHFCMzFSIyUYGhJkFCRFNxkf/aAAgBAQABPwCz0WtG3Vnw5rZf"
    "Ln6dWfA+tl8n7ddwO35fcobalhUAF9YyCDMpTvztCv8AR9MF0Otk2nrBeHNbJ5f26u98smEahFuFE5c7qUBZpmlBKWgSI1F75SUk"
    "kAAnYTAIJu14xvjTENVWe0lUFK4+FMtW5BpwEpTkSc3qGQJIUoiTwIAHa6i1tLaeOr/dnsI4bpBbkNm63BZbpitJIaSB77kRBKZS"
    "AD+agYIBBs+HWbXhSquD6m6dKGiA4twNhEjdWY8ZRKv6SeOemb7dbtVrtuEkeCtvwCsDcVDwghRST6aTIiAFjKDInKOyVFraG08d"
    "PsLxF3MrVZXCzQrFCylaEgpDZIWNuQXNQgneCOOB35rqq74gocAWt5QtdGEv16WwnLUPycqSoEkhG4y7QuZBKUw7UeC/DGGDNf6d"
    "bWt/t/m2g/yfNX6eB73w/wD/xAAoEQACAAMGBQUAAAAAAAAAAAABAgMEUQAGESExQQUSFCJhMnGRocH/2gAIAQIBAT8AvvxbpubO"
    "0zPTs1FaPiEQDEE740GvnYUtfVljTriJ6EBJ80Hz9A2nVZoJjxe5nzAOirsW/Peun//EACURAQABAgQGAwEAAAAAAAAAAAECBREA"
    "AwQhBjJBUWGREhQiMf/aAAgBAwEBPwDQ6z6dscT16owo0YQkZcc5AZNlibqBu9DoI/3FP0cNRUbZvJA+Sd+x79gmOJ9VOp1sjmfp"
    "hEsPLG+7KXncLeNhXH//2Q=="
)


# Layouts Pillow never writes, from cjpeg, which libjpeg decodes (tests/test_image_diff.py): vertical-only
# 2:1 chroma (4:4:0, -sample 1x2,1x1,1x1), decoded with replicated rows up to 69 levels off, and SOF1
# (-quality 1: 16-bit quantizers), which was refused.
CHROMA_440_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0d"
    "Hx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh7/wAARCAARAA8DARIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm"
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCX4a/8U19t/wCGaf8AitftPl/29/bf7r7Lt3fZ/L3fZ87t0+cb/uL93uWn/FUY/wCbdfs3"
    "/bn/AG3u/wDAbf5O3/bx5/8ADn5sahhVPKFjsNP1jUYvAE8mraf9odRLdja3lBj5TchOSMk8fgKk1mT+0fEGoy/2d/wgv+lSt9i2"
    "+VjLn91jEf8Aq+mMcZ6CvscBeFCPsbp2V+RqD2+06mkn5w0Wt90fe5Y5Qw0Hh7puMb+zapt6fadXSb3s4aLW+6PU/wBvP/mSf+3/"
    "AP8Abeivjqh8FVPO/wBoT/koOqf9hW+/9G0V9PxT/wAi/Af4P/bYH2HGn/Iryz/r3/7bTP/Z"
)
SOF1_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCDEAMgAiYCWAK8AlgB9AMgArwCigK8A4QDUgMgA7YEsAfQBRQEsARMBEwEsAmSBtYHOgWq"
    "B9ALVAn2C+oLuAsiCfYK8Aq+DIAOEBH4DzwMgA1IEP4Negq+CvAPoBVKD9IQ/hKOEyQUHhRQFB4MHA8KFhIXohXgE4gXcBH4E7oU"
    "HhNW/9sAgxEDUgOEA4QEsAQaBLAJLgUUBRQJLhNWDOQK8AzkE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YT"
    "VhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVhNWE1YTVv/BABEIABAAEAMBIgAC"
    "EQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEH"
    "InEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOE"
    "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAf"
    "AQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGh"
    "scEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqS"
    "k5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/"
    "ACiiigD/2Q=="
)


# A progressive file from tests/fuzz_jpeg_diff.c whose luma AC scan holds "FF FF FF 00": libjpeg's slow
# path read it as one FF data byte while the old decoder ended the scan there and decoded the rest
# from padding. Fill before a stuffed zero is not standard and is now refused (see with_padded_stuffing).
FF_RUN_BEFORE_STUFFED_ZERO_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0d"
    "Hx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh7/wgARCAAgACADASIAAhEBAxEB/8QAGQABAQADAQAAAAAAAAAAAAAABgcDBAUI/9oACAEBAAAAAPPjJnpDbcvhmK1n"
    "533l5/gf/8QAFQEBAQAAAAAAAAAAAAAAAAAABwj/2gAIAQIQAAAAbJ6WyX//xAAVAQEBAAAAAAAAAAAAAAAAAAAFBP/aAAgBAxAA"
    "AAChcov/xAAwEAABAwIFAgMGBwAAAAAAAAABAgMRBAUGEhMhMQAHFCMzFSIyUYGhJkFCRFNxkf/aAAgBAQABPwCz0WtG3Vnw5rZf"
    "Ln6dWfA+tl8n7ddwO37///8AbS7SWdYyCDMpTvztCv8AR9MF0Otk2nrBeHNbGCeX9urvfLJhGoRbhROXO6lAWaZpQSloEiNRguUl"
    "JJAAJ2EwCCbteMb40xDVVntJVBSuPhTLVuQacBKU5EnN6hkCSFKIk8CAB2uotbS2njq/3Z7COG6QW5DZutwWW6YrSSGkge+5EQSm"
    "UgA/moGCAQbPh1m14Uqrg+punShogOLcDYRI3VmPGUSr+knjnpm+3W7Va7bhJHgrb8ArA3FQ8IIUUiymkyIgBYygyJyjslRa2htP"
    "HT7C8RdzK1WVws0KxQspWhIKQ2SFjbkFzUIJ3gjjgd+a6qu+IKHAFreULXRhL9elsJw1D8nKkqBJIRuMu0LmQSlMO1HgvwxhgzX+"
    "nW1rf7f5toP8nzV+nge98P8A/8QAKBEAAgADBgUFAAAAAAAAAAAAAQIDBFEABhEhMUEFEhQiYTJxkaHB/9oACAECAQE/AL78W6bm"
    "ztMz07NRWj4hEAxBO+NBr52FLX1ZY064iehASfNB8/QNp1WaCY8XuZ8wDoq7Fvz3rp//xAAlEQEAAQIEBgMBAAAAAAAAAAABAgUR"
    "AAMEIQYyQVFhkRIUIjH/2gAIAQMBAT8A0Os+nbHE9eqMKNGEJGXHOQGTZYm6gbvQ6CP9xT9HDUVG2byQPknfse/YJjifVTqdbI5n"
    "6YRLDyxvuyl53C3jYVx//9k="
)


# An 8x8 one-pixel checkerboard from libjpeg-turbo 3.2's `cjpeg -quality 3` (16-bit quantizers, SOF1).
# Its IDCT output reaches 397 before the level shift, the largest measured over 3,692 encoder files;
# the decoder refuses blocks beyond [-512, 511], where libjpeg-turbo's C and NEON paths disagree.
CJPEG_Q3_CHECKER_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCDEAELALcAyADpAMgApwELAOkA2QDpASwBGwELAT0BkAKaAbEBkAFvAW8BkAMwAkcCaAHj"
    "ApoDxgNSA/gD6AO2A1IDpQOUBCoEsAX9BRMEKgRtBakEfgOUA6UFNQcYBUUFqQYvBmEGtAbFBrQECQUDB1sH4AdKBoIHzwX9BpMG"
    "tAZx/9sAgxEBGwEsASwBkAFeAZADDwGxAbEDDwZxBEwDpQRMBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEG"
    "cQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcQZxBnEGcf/BABEIAAgACAMBIgAC"
    "EQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEH"
    "InEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOE"
    "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAf"
    "AQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGh"
    "scEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqS"
    "k5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/"
    "AD/P+f8AP/1iiigD/9k="
)


# A 7x3 progressive JPEG found by tests/fuzz_jpeg_diff.c: in Cr's first AC scan (Ss = 1, Se = 63) a run
# passes position 63 at the end of the data. libjpeg writes that coefficient at 63 through the
# padding of its natural order; the decoder dropped it, 29 of 63 values off by up to 18 (review #3).
AC_FIRST_OVERSHOOT_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoK"
    "CgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoK"
    "CgoKCgoKCgr/wgARCAADAAcDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABUBAQEAAAAAAAAAAAAAAAAAAAYI/9oA"
    "DAMBAAIQAxAAAAEuEqL/AP/EABYQAQEBAAAAAAAAAAAAAAAAAAIGE//aAAgBAQABBQKOBz//xAAaEQABBQEAAAAAAAAAAAAAAABB"
    "ZKJiZSFB/9oACAEDAQE/AYzEOls4v//EABgRAAIDAAAAAAAAAAAAAAAAAAACBSEi/9oACAECAQE/AZfTWf/EABYQAAMAAAAAAAAA"
    "AAAAAAAAAAAxQf/aAAgBAQAGPwJQ/8QAFhAAAwAAAAAAAAAAAAAAAAAAABEh/9oACAEBAAE/IU0n/9oADAMBAAIAAwAAABBf/8QA"
    "FhEBAQEAAAAAAAAAAAAAAAAAMQCx/9oACAEDAQE/ECZZf//EABYRAAMAAAAAAAAAAAAAAAAAAAAhYf/aAAgBAgEBPxCgz//EABcQ"
    "AAMBAAAAAAAAAAAAAAAAAAARMcH/2gAIAQEAAT8QdZYP/9k="
)


# A 24x16 sequential (SOF0) JPEG with 4:2:0 chroma and one scan per component, from libjpeg-turbo
# 3.2's `cjpeg -quality 85 -sample 2x2,1x1,1x1 -scans` with the script "0; 1; 2;". Pillow decodes it;
# the decoder required every component in one scan and refused it (review #3, Codex on a82292d).
SEQUENTIAL_SCAN_PER_COMPONENT_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0d"
    "Hx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh7/wAARCAAQABgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AE8Qf8TP/hJ/+Fl/8ST+2vsn/C2PsPzf2D5OP7H+x48zf5+I/M2/aMZOfK7HiD/i"
    "Z/8ACT/8LL/4kn9tfZP+FsfYfm/sHycf2P8AY8eZv8/EfmbftGMnPldjxB/xM/8AhJ/+Fl/8ST+2vsn/AAtj7D839g+Tj+x/sePM"
    "3+fiPzNv2jGTnyux/wAgv/qov9h/9tv+Fn+f/wB/PP8A7Nz/ANPOzy/+WPY/5Bf/AFUX+w/+23/Cz/P/AO/nn/2bn/p52eX/AMse"
    "x/yC/wDqov8AYf8A22/4Wf5//fzz/wCzc/8ATzs8v/lj2//EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIB"
    "AgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVG"
    "R0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT"
    "1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAgBAhEAPwAA/9oACAEDEQA/AAD/2Q=="
)


def jpeg_segments(jpeg: bytes) -> list[tuple[int, bytes]]:
    """(marker, bytes) for each segment from SOI to EOI, entropy-coded data attached to its SOS."""
    segs, pos = [(0xD8, jpeg[:2])], 2
    while pos < len(jpeg):
        assert jpeg[pos] == 0xFF
        m = jpeg[pos + 1]
        if m == 0xD9:
            segs.append((m, jpeg[pos:pos + 2]))
            break
        end = pos + 2 + int.from_bytes(jpeg[pos + 2:pos + 4], "big")
        if m == 0xDA:
            while end + 1 < len(jpeg) and not (jpeg[end] == 0xFF and jpeg[end + 1] != 0x00 and not 0xD0 <= jpeg[end + 1] <= 0xD7):
                end += 1
        segs.append((m, jpeg[pos:end]))
        pos = end
    return segs


def dqt_tables(jpeg: bytes) -> dict[int, bytes]:
    """The 8-bit quantization tables a JPEG defines, by table id."""
    tables = {}
    for m, seg in jpeg_segments(jpeg):
        off = 4
        while m == 0xDB and off < len(seg):
            assert seg[off] >> 4 == 0
            tables[seg[off] & 15] = seg[off + 1:off + 65]
            off += 65
    return tables


def dqt(table_id: int, values: bytes) -> bytes:
    return bytes([0xFF, 0xDB, 0, 67, table_id]) + values


def gray_block_jpeg(ac_symbols: list[int], scan_bits: str) -> bytes:
    """One 8x8 gray block (SOF1, every quantizer 1). DC symbols 0-15 take 5-bit codes and the given
    AC symbols 8-bit codes, in order; scan_bits is the entropy-coded data as a bit string."""
    def dht(th: int, symbols: list[int], length: int) -> bytes:
        counts = [0] * 16
        counts[length - 1] = len(symbols)
        body = bytes([th, *counts, *symbols])
        return b"\xff\xc4" + (len(body) + 2).to_bytes(2, "big") + body
    scan_bits += "1" * (-len(scan_bits) % 8)
    data = bytes(int(scan_bits[i:i + 8], 2) for i in range(0, len(scan_bits), 8)).replace(b"\xff", b"\xff\x00")
    return (b"\xff\xd8" + dqt(0, bytes([1] * 64)) + bytes([0xFF, 0xC1, 0, 11, 8, 0, 8, 0, 8, 1, 1, 0x11, 0]) +
            dht(0x00, list(range(16)), 5) + dht(0x10, ac_symbols, 8) + bytes([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0]) +
            data + b"\xff\xd9")


def progressive_gray_block_jpeg(ac_symbols: list[int], scans: list[tuple[int, int, str]]) -> bytes:
    """One 8x8 gray block, progressive (SOF2), every quantizer 1: a DC first scan coding a zero
    difference, then one AC first scan per (Ss, Se, scan_bits), all with Al = 0. Tables as in
    gray_block_jpeg."""
    base = gray_block_jpeg(ac_symbols, "00000")
    head = base[:base.index(b"\xff\xda")].replace(bytes([0xFF, 0xC1]), bytes([0xFF, 0xC2]), 1)
    def scan(ss: int, se: int, bits: str) -> bytes:
        bits += "1" * (-len(bits) % 8)
        data = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)).replace(b"\xff", b"\xff\x00")
        return bytes([0xFF, 0xDA, 0, 8, 1, 1, 0, ss, se, 0]) + data
    return head + scan(0, 0, "00000") + b"".join(scan(*x) for x in scans) + b"\xff\xd9"


def single_coefficient_jpeg(index: int, value: int) -> bytes:
    """gray_block_jpeg holding one coefficient at zigzag index 1-15 (or DC at 0), with whatever
    category its value needs: libjpeg accepts AC categories up to 15, not only the 10 of 8-bit data."""
    n = abs(value).bit_length()
    extra = format(value if value >= 0 else value + (1 << n) - 1, f"0{n}b") if n else ""
    if index == 0:
        return gray_block_jpeg([0x00], format(n, "05b") + extra + "0" * 8)
    sym = ((index - 1) << 4) | n
    return gray_block_jpeg([0x00, sym], "00000" + format(1, "08b") + extra + "0" * 8)


def with_fill_bytes(jpeg: bytes) -> bytes:
    """The same JPEG with one 0xFF fill byte before every marker after SOI, which the format allows
    and libjpeg skips. The vendored decoder stepped over "FF FF" as a pair and so lost the marker's
    own FF (review #3, found by tests/fuzz_jpeg_diff.c)."""
    out, pos = bytearray(jpeg[:2]), 2
    while pos < len(jpeg):
        assert jpeg[pos] == 0xFF
        m = jpeg[pos + 1]
        out += b"\xff"
        if m == 0xD9:
            out += jpeg[pos:]
            break
        seg = int.from_bytes(jpeg[pos + 2:pos + 4], "big")
        end = pos + 2 + seg
        if m == 0xDA:   # entropy-coded data runs to the next marker that is not stuffing or RSTn
            while end + 1 < len(jpeg) and not (jpeg[end] == 0xFF and jpeg[end + 1] not in (0x00,) and not 0xD0 <= jpeg[end + 1] <= 0xD7):
                end += 1
        out += jpeg[pos:end]
        pos = end
    return bytes(out)


def with_padded_stuffing(jpeg: bytes) -> bytes:
    """Every stuffed FF 00 in the entropy-coded data written as FF FF 00, fill before a stuffed zero,
    which is not standard JPEG. libjpeg's slow path reads it as one FF data byte, but libjpeg-turbo's
    result for it was measured to depend on how its input is buffered (docs/vision.md), so the engine
    refuses it (review #3, found by tests/fuzz_jpeg_diff.c)."""
    sos = jpeg.index(b"\xff\xda")
    head, body = jpeg[:sos], jpeg[sos:]
    seg = int.from_bytes(body[2:4], "big")
    return head + body[:2 + seg] + body[2 + seg:].replace(b"\xff\x00", b"\xff\xff\x00")


def with_colour_markers(jpeg: bytes, *, jfif: bool, adobe: int | None = None, ids: bytes = b"\x01\x02\x03",
                        adobe_after_first_scan: bool = False) -> bytes:
    """A three-component Pillow JPEG with its colour-space signals rewritten: the JFIF APP0 kept or
    dropped, an Adobe APP14 with the given transform added (before the frame, or after the first
    scan), and the component ids in the frame and every scan header replaced. The entropy-coded data
    is unchanged, so libjpeg (and so Pillow) either converts the planes from YCbCr or copies them as
    RGB, by jdapimin.c default_decompress_parms; the vendored decoder converted them in every case
    (review #3, Codex on b1e7dd2)."""
    remap = dict(zip(b"\x01\x02\x03", ids))
    app14 = None if adobe is None else b"\xff\xee\x00\x0eAdobe\x00\x64\x00\x00\x00\x00" + bytes([adobe])
    out, pos, scans = bytearray(jpeg[:2]), 2, 0
    if app14 and not adobe_after_first_scan:
        out += app14
    while pos < len(jpeg):
        assert jpeg[pos] == 0xFF
        m = jpeg[pos + 1]
        if m == 0xD9:
            out += jpeg[pos:]
            break
        end = pos + 2 + int.from_bytes(jpeg[pos + 2:pos + 4], "big")
        seg = bytearray(jpeg[pos:end])
        if m == 0xDA:
            if app14 and adobe_after_first_scan and scans == 1:
                out += app14
            scans += 1
            for c in range(seg[4]):
                seg[5 + 2 * c] = remap[seg[5 + 2 * c]]
            while end + 1 < len(jpeg) and not (jpeg[end] == 0xFF and jpeg[end + 1] not in (0x00,) and not 0xD0 <= jpeg[end + 1] <= 0xD7):
                end += 1
            seg += jpeg[pos + len(seg):end]
        elif m in (0xC0, 0xC2):
            assert seg[9] == 3
            for c in range(3):
                seg[10 + 3 * c] = remap[seg[10 + 3 * c]]
        if not (m == 0xE0 and seg[4:9] == b"JFIF\0" and not jfif):
            out += seg
        pos = end
    if adobe_after_first_scan and scans < 2:
        raise AssertionError("adobe_after_first_scan needs a file with at least two scans")
    return bytes(out)


def progressive_se255(arr: np.ndarray) -> bytes:
    """A progressive JPEG whose first AC scan declares Se = 255: the coefficient index ran past the
    64-entry zigzag table (global over-read feeding a heap write) until the scan header was validated."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90, progressive=True)
    data = bytearray(buf.getvalue())
    pos = 2
    while pos + 4 <= len(data):
        marker, seg = data[pos + 1], int.from_bytes(data[pos + 2:pos + 4], "big")
        if marker == 0xDA:
            ns = data[pos + 4]
            ss_off = pos + 5 + ns * 2
            if data[ss_off] != 0:   # the first AC scan
                data[ss_off + 1] = 0xFF
                return bytes(data)
            pos += 2 + seg
            while pos + 1 < len(data) and not (data[pos] == 0xFF and data[pos + 1] not in (0, *range(0xD0, 0xD8))):
                pos += 1
        else:
            pos += 2 + seg
    raise AssertionError("no AC scan found")


def processor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    return Qwen2VLImageProcessor(min_pixels=min_pixels, max_pixels=max_pixels, patch_size=16, merge_size=2,
                                 temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)


def synth(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = rng.integers(0, 4)
    y, x = np.indices((h, w))
    if kind == 0:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if kind == 1:
        return np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1).astype(np.uint8)
    if kind == 2:
        img = np.full((h, w, 3), 240, np.uint8)
        for _ in range(6):
            y0, y1 = sorted(rng.integers(0, h + 1, 2))
            x0, x1 = sorted(rng.integers(0, w + 1, 2))
            img[y0:y1, x0:x1] = rng.integers(0, 256, 3)
        return img
    return np.where((x // 2 + y // 3) % 2 == 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)


def encode(img: Image.Image, fmt: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def png_stream(stream: bytes, width: int = 1, height: int = 1) -> str:
    """Wrap a controlled zlib stream in CRC-valid RGB PNG chunks."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", stream) + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def run_tool(lines: list[str]) -> list[str]:
    out = subprocess.run([ROOT / "clef-tool", "image"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout
    return out.split("\n")[: len(lines)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cases", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="clef-image-") as tmp:
        cases = []   # (name, b64, expected PIL image, min_pixels, max_pixels)
        # decoder coverage: PNG color types, JPEG progressive/subsampling/grayscale, sizes
        for i in range(args.cases):
            h, w = (int(rng.integers(1, 97)), int(rng.integers(1, 97))) if i % 3 else (int(rng.integers(100, 700)), int(rng.integers(100, 700)))
            arr = synth(rng, h, w)
            base = Image.fromarray(arr)
            choice = i % 7
            if choice == 0:
                im, b64 = base, encode(base, "PNG")
            elif choice == 1:
                rgba = np.concatenate([arr, rng.integers(0, 256, (h, w, 1), dtype=np.uint8)], -1)
                im, b64 = Image.fromarray(rgba, "RGBA"), encode(Image.fromarray(rgba, "RGBA"), "PNG")
            elif choice == 2:
                gray = base.convert("L")
                im, b64 = gray, encode(gray, "PNG")
            elif choice == 3:
                la = Image.fromarray(np.concatenate([np.asarray(base.convert("L"))[..., None], rng.integers(0, 256, (h, w, 1), dtype=np.uint8)], -1), "LA")
                im, b64 = la, encode(la, "PNG")
            elif choice == 4:
                pal = base.convert("P", palette=Image.ADAPTIVE, colors=int(rng.integers(2, 256)))
                im, b64 = pal, encode(pal, "PNG")
            elif choice == 5:
                b64 = encode(base, "JPEG", quality=int(rng.integers(50, 100)), subsampling=int(rng.integers(0, 3)),
                             progressive=bool(rng.integers(0, 2)))
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
            else:
                b64 = encode(base.convert("L"), "JPEG", quality=85, progressive=bool(rng.integers(0, 2)))
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
            mn, mx = MIN_PIXELS, MAX_PIXELS
            if i % 5 == 4:   # media_kwargs-style bounds: force down- and up-scaling paths
                mn, mx = int(rng.choice([1024, 65536, 200000])), int(rng.choice([65536, 102400, 1048576]))
                mn = min(mn, mx)
            cases.append((f"case{i}", b64 if i % 2 else ("DATA:" if i % 4 else "data:") + "image/x;base64," + b64, im, mn, mx))
        # Exercise all DEFLATE block types independently of Pillow's compression defaults.
        arr = synth(np.random.default_rng(19), 256, 256)
        raw = b"".join(b"\0" + row.tobytes() for row in arr)
        for name, level, strategy in [("stored", 0, zlib.Z_DEFAULT_STRATEGY),
                                       ("fixed", 6, zlib.Z_FIXED), ("dynamic", 6, zlib.Z_DEFAULT_STRATEGY)]:
            compressor = zlib.compressobj(level, zlib.DEFLATED, zlib.MAX_WBITS, 8, strategy)
            stream = compressor.compress(raw) + compressor.flush()
            cases.append((name, png_stream(stream, 256, 256), Image.fromarray(arr), MIN_PIXELS, MAX_PIXELS))
        # Empty IDAT chunks, first and between data chunks: legal, and skipped (Codex on 600ddfe: the
        # accumulation reallocated to zero bytes, which C lets return NULL).
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        arr = synth(np.random.default_rng(59), 6, 8)
        z = zlib.compress(b"".join(b"\0" + row.tobytes() for row in arr))
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 6, 8, 2, 0, 0, 0))
        for name, idats in (("png-empty-first-idat", [b"", z]), ("png-empty-middle-idat", [z[:10], b"", z[10:]])):
            data = b"\x89PNG\r\n\x1a\n" + ihdr + b"".join(chunk(b"IDAT", x) for x in idats) + chunk(b"IEND", b"")
            cases.append((name, base64.b64encode(data).decode(), Image.open(io.BytesIO(data)), MIN_PIXELS, MAX_PIXELS))
        for name, b64 in (("separate-dc-37x21", SEPARATE_DC_SCANS_37X21), ("separate-dc-32x32", SEPARATE_DC_SCANS_32X32),
                          ("chroma-440", CHROMA_440_JPEG), ("sof1", SOF1_JPEG)):
            cases.append((name, b64, Image.open(io.BytesIO(base64.b64decode(b64))), MIN_PIXELS, MAX_PIXELS))
        for prog in (False, True):
            buf = io.BytesIO()
            Image.fromarray(synth(np.random.default_rng(23), 37, 53)).save(buf, "JPEG", quality=80, progressive=prog)
            filled = with_fill_bytes(buf.getvalue())
            cases.append((f"fill-bytes{'-prog' if prog else ''}", base64.b64encode(filled).decode(),
                          Image.open(io.BytesIO(filled)), MIN_PIXELS, MAX_PIXELS))
        # A one-component frame declaring 2x2 sampling, which libjpeg and Pillow ignore: the old decoder
        # walked its blocks as 2x2 MCUs (review #3, a 2x1 frame found by tests/fuzz_jpeg_diff.c).
        gray = io.BytesIO()
        Image.fromarray(synth(np.random.default_rng(31), 21, 37)).convert("L").save(gray, "JPEG", quality=85)
        g = bytearray(gray.getvalue())
        sof = g.index(b"\xff\xc0")
        if g[sof + 9] != 1 or g[sof + 11] != 0x11:
            sys.exit("gray-sampling fixture is not a one-component 1x1 frame")
        g[sof + 11] = 0x22
        cases.append(("gray-declared-2x2", base64.b64encode(bytes(g)).decode(), Image.open(io.BytesIO(bytes(g))), MIN_PIXELS, MAX_PIXELS))
        # Three-component colour space, as libjpeg decides it: JFIF means YCbCr, otherwise an Adobe
        # transform of 0 means RGB and any other YCbCr, otherwise ids 'R','G','B' mean RGB. The first
        # two rows of each group are RGB-coded and decoded wrong before review #3's fix; the rest are
        # YCbCr controls, including an RGB transform that arrives only after the first scan.
        rgb_ids = b"RGB"
        for prog, sub in ((False, 0), (True, 2)):
            buf = io.BytesIO()
            Image.fromarray(synth(np.random.default_rng(37), 29, 43)).save(buf, "JPEG", quality=85, subsampling=sub, progressive=prog)
            variants = [("rgb-ids", dict(jfif=False, ids=rgb_ids)), ("adobe0", dict(jfif=False, adobe=0)),
                        ("adobe1-rgb-ids", dict(jfif=False, adobe=1, ids=rgb_ids)),
                        ("jfif-adobe0-rgb-ids", dict(jfif=True, adobe=0, ids=rgb_ids)),
                        ("adobe2", dict(jfif=False, adobe=2)), ("other-ids", dict(jfif=False, ids=b"RGC"))]
            if prog:
                variants.append(("adobe0-after-scan", dict(jfif=False, adobe=0, adobe_after_first_scan=True)))
            for vname, kw in variants:
                data = with_colour_markers(buf.getvalue(), **kw)
                cases.append((f"colour-{vname}{'-prog' if prog else ''}", base64.b64encode(data).decode(),
                              Image.open(io.BytesIO(data)), MIN_PIXELS, MAX_PIXELS))
        # Quantization tables are latched per component at its first scan, as libjpeg does: a table
        # redefined after the first scan of a progressive file must not change it, and one defined
        # between the frame and the scan is in time (review #3, Codex on 3f3eb03). AC categories above
        # 10 and run/size symbols with size 0 decode as libjpeg decodes them; the IDCT output at 511
        # and the cjpeg quality-3 checkerboard (397) stay inside the decoder's range.
        prog = io.BytesIO()
        Image.fromarray(synth(np.random.default_rng(43), 16, 24)).save(prog, "JPEG", quality=80, progressive=True)
        segs = jpeg_segments(prog.getvalue())
        first_sos = next(i for i, (m, _) in enumerate(segs) if m == 0xDA)
        redefined = b"".join(s for _, s in segs[:first_sos + 1]) + dqt(0, bytes([1] * 64)) + b"".join(s for _, s in segs[first_sos + 1:])
        base = io.BytesIO()
        Image.fromarray(synth(np.random.default_rng(47), 16, 16)).convert("L").save(base, "JPEG", quality=75)
        segs = [x for x in jpeg_segments(base.getvalue()) if x[0] != 0xDB]
        sof = next(i for i, (m, _) in enumerate(segs) if m == 0xC0)
        late_dqt = b"".join(s for _, s in segs[:sof + 1]) + dqt(0, dqt_tables(base.getvalue())[0]) + b"".join(s for _, s in segs[sof + 1:])
        for name, data in (("dqt-redefined-after-scan", redefined), ("dqt-after-frame", late_dqt),
                           ("ac-category-11", single_coefficient_jpeg(1, 1024)),
                           ("ac-run3-size0", gray_block_jpeg([0x30], "00000" + "0" * 8)),
                           ("idct-511", single_coefficient_jpeg(0, 4088)),
                           ("cjpeg-q3-checker", base64.b64decode(CJPEG_Q3_CHECKER_JPEG)),
                           # Runs that pass the band's end: libjpeg writes the coefficient at zigzag
                           # position k, or 63 past the table, and ends the band, without a warning.
                           # The decoder dropped it at the end of the data and refused the file
                           # elsewhere (tests/fuzz_jpeg_diff.c).
                           ("ac-first-overshoot-fuzz", base64.b64decode(AC_FIRST_OVERSHOOT_JPEG)),
                           ("ac-first-overshoot-band", progressive_gray_block_jpeg(
                               [0x00, 0xA1], [(1, 5, "00000001" + "1"), (6, 63, "00000000")])),
                           ("baseline-overshoot-63", gray_block_jpeg([0x00, 0xF1], "00000" + ("00000001" + "1") * 4 + "00000000"))):
            cases.append((name, base64.b64encode(data).decode(), Image.open(io.BytesIO(data)), MIN_PIXELS, MAX_PIXELS))
        # A sequential frame split across scans is buffered like a progressive one. libjpeg decodes a
        # component that is never scanned as zero coefficients and keeps the first scan's values where
        # a second scan of a component codes zeros, both without a warning.
        seq = base64.b64decode(SEQUENTIAL_SCAN_PER_COMPONENT_JPEG)
        segs = jpeg_segments(seq)
        sos = [i for i, (m, _) in enumerate(segs) if m == 0xDA]
        if len(sos) != 3 or next(s for m, s in segs if m == 0xC0)[9] != 3:
            sys.exit("sequential fixture: expected a three-component SOF0 frame with three scans")
        for name, data in (("seq-scan-per-component", seq),
                           ("seq-cr-never-scanned", b"".join(s for i, (_, s) in enumerate(segs) if i != sos[2])),
                           ("seq-cb-scanned-twice", b"".join(s for _, s in segs[:-1]) + segs[sos[1]][1] + b"\xff\xd9")):
            cases.append((name, base64.b64encode(data).decode(), Image.open(io.BytesIO(data)), MIN_PIXELS, MAX_PIXELS))
        lines = [json.dumps({"image": b64, "out": f"{tmp}/{name}", "min_pixels": mn, "max_pixels": mx})
                 for name, b64, _, mn, mx in cases]
        results = run_tool(lines)
        for (name, _, im, mn, mx), got in zip(cases, results):
            ref_rgb = np.asarray(im.convert("RGB"))
            if got.startswith("ERR"):
                print(f"{name}: engine error: {got}")
                failures += 1
                continue
            meta = json.loads(got)
            H, W = ref_rgb.shape[:2]
            rh, rw = smart_resize(H, W, factor=32, min_pixels=mn, max_pixels=mx)
            if (meta["decoded"], meta["width"], meta["height"]) != ([W, H], rw, rh):
                print(f"{name}: size mismatch {meta} vs decoded {W}x{H} resized {rw}x{rh}")
                failures += 1
                continue
            dec = np.fromfile(f"{tmp}/{name}.rgb", np.uint8).reshape(H, W, 3)
            if not np.array_equal(dec, ref_rgb):
                print(f"{name}: decoded pixels differ ({int((dec != ref_rgb).sum())} values, max {int(np.abs(dec.astype(int) - ref_rgb).max())})")
                failures += 1
                continue
            t = tvF.pil_to_tensor(im.convert("RGB"))
            ref_resized = tvF.resize(t, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
            got_resized = np.fromfile(f"{tmp}/{name}.resized", np.uint8).reshape(rh, rw, 3)
            if not np.array_equal(got_resized, ref_resized):
                d = np.abs(got_resized.astype(int) - ref_resized.astype(int))
                print(f"{name}: resized pixels differ ({W}x{H} -> {rw}x{rh}: {int((d > 0).sum())} values, max {int(d.max())})")
                failures += 1
                continue
            out = processor(mn, mx)(images=im, return_tensors="pt")
            ref_patches = out["pixel_values"].numpy()
            got_patches = np.fromfile(f"{tmp}/{name}.patches", np.float32).reshape(ref_patches.shape[0], -1)
            grid = out["image_grid_thw"][0].tolist()
            if grid != [1, meta["grid_h"], meta["grid_w"]] or got_patches.shape != ref_patches.shape or not np.array_equal(got_patches, ref_patches):
                print(f"{name}: patches differ (grid {grid} vs {meta}, max abs {float(np.abs(got_patches - ref_patches).max()) if got_patches.shape == ref_patches.shape else 'shape'})")
                failures += 1
                continue
            n = ref_patches.shape[0]
            raw = np.fromfile(f"{tmp}/{name}.pos", np.uint8)
            got_idx = raw[: n * 16].view(np.int32).reshape(n, 4)
            got_w = raw[n * 16:].view(np.float32).reshape(n, 4)
            ref_idx, ref_w = get_vision_bilinear_indices_and_weights(torch.tensor([grid]), 48, 2)
            ref_idx, ref_w = ref_idx.numpy().T, ref_w.numpy().T
            if not np.array_equal(got_idx, ref_idx) or not np.allclose(got_w, ref_w, rtol=0, atol=2e-7):
                print(f"{name}: position interpolation differs (idx equal {np.array_equal(got_idx, ref_idx)}, max |dw| {float(np.abs(got_w - ref_w).max())})")
                failures += 1
                continue
        print(f"images: {len(cases) - failures}/{len(cases)} byte-identical through decode, resize, patches and positions")

        # error handling: bad base64, truncated image, aspect ratio, unsupported bit depth, and the
        # two crafted JPEGs that overflowed the vendored decoder before its scan-header and sampling
        # checks (ASan-confirmed on the unmodified copy; both must be rejected, never decoded)
        buf = io.BytesIO(); Image.fromarray(np.zeros((4, 4), np.uint16)).save(buf, "PNG")
        wide = Image.fromarray(np.zeros((2, 500, 3), np.uint8))
        se255 = progressive_se255(np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8))
        bad = [json.dumps({"image": "abc", "out": f"{tmp}/bad0"}),
               json.dumps({"image": base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode(), "out": f"{tmp}/bad1"}),
               json.dumps({"image": encode(wide, "PNG"), "out": f"{tmp}/bad2"}),
               json.dumps({"image": base64.b64encode(buf.getvalue()).decode(), "out": f"{tmp}/bad3"}),
               json.dumps({"image": "data:image/png,notbase64", "out": f"{tmp}/bad4"}),
               json.dumps({"image": base64.b64encode(se255).decode(), "out": f"{tmp}/bad5"}),
               json.dumps({"image": LUMA_UNDER_CHROMA_JPEG, "out": f"{tmp}/bad6"})]
        noisy = io.BytesIO()
        Image.fromarray(np.random.default_rng(29).integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(noisy, "JPEG", quality=95)
        padded = with_padded_stuffing(noisy.getvalue())
        if padded.count(b"\xff\xff\x00") < 3:
            sys.exit("padded-stuffing fixture has too few stuffed bytes to test anything")
        bad.append(json.dumps({"image": base64.b64encode(padded).decode(), "out": f"{tmp}/bad7"}))
        bad.append(json.dumps({"image": FF_RUN_BEFORE_STUFFED_ZERO_JPEG, "out": f"{tmp}/bad8"}))
        # A valid progressive file cut before its first scan and closed with EOI: libjpeg fails with
        # JERR_SOF_NO_SOS ("missing SOS marker") and Pillow refuses it, but the vendored decoder
        # returned a blank image (review #3, Codex on b1e7dd2). The baseline cut was already refused
        # and stays a control.
        for i, prog in enumerate((True, False)):
            cut = io.BytesIO()
            Image.fromarray(synth(np.random.default_rng(41), 24, 40)).save(cut, "JPEG", quality=85, progressive=prog)
            data = cut.getvalue()
            bad.append(json.dumps({"image": base64.b64encode(data[:data.index(b"\xff\xda")] + b"\xff\xd9").decode(),
                                   "out": f"{tmp}/bad-no-scan{i}"}))
        # A quantization table that is not defined when its component's first scan starts: libjpeg
        # fails with JERR_NO_QUANT_TABLE, and the decoder used the zero-filled slot, a flat gray image.
        # And blocks whose IDCT output leaves [-512, 511]: libjpeg-turbo's C path wraps 512 to black
        # where its NEON path (Pillow on this Mac) clamps, and a category-15 AC coefficient decoded
        # 128 levels from Pillow (review #3, Codex on 3f3eb03). 513 is refused although Pillow here
        # clamps it: no wider range is stable across libjpeg builds.
        gray = base.getvalue()
        segs = jpeg_segments(gray)
        no_dqt = b"".join(s for m, s in segs if m != 0xDB)
        wrong_id = b"".join(dqt(1, dqt_tables(gray)[0]) if m == 0xDB else s for m, s in segs)
        after_scan = b"".join(s for m, s in segs if m not in (0xDB, 0xD9)) + dqt(0, bytes([1] * 64)) + b"\xff\xd9"
        colour = io.BytesIO()
        Image.fromarray(synth(np.random.default_rng(53), 16, 24)).save(colour, "JPEG", quality=80, progressive=True)
        tables, segs = dqt_tables(colour.getvalue()), [x for x in jpeg_segments(colour.getvalue()) if x[0] != 0xDB]
        first_sos = next(i for i, (m, _) in enumerate(segs) if m == 0xDA)
        sof = next(s for m, s in segs if m == 0xC2)
        if 1 not in tables or segs[first_sos][1][4] != 3 or sof[9] != 3 or (sof[12], sof[15], sof[18]) != (0, 1, 1):
            sys.exit("late-table fixture: expected an interleaved first scan of three components, chroma on table 1")
        late_table1 = (segs[0][1] + dqt(0, tables[0]) + b"".join(s for _, s in segs[1:first_sos + 1]) + dqt(1, tables[1]) +
                       b"".join(s for _, s in segs[first_sos + 1:]))
        # Frame-header order, as libjpeg reads it: a scan before the frame header (JERR_SOS_NO_SOF), and
        # a second frame header (JERR_SOF_DUPLICATE); the decoder found the frame by looking ahead and
        # decoded both (Codex on a82292d).
        segs = jpeg_segments(gray)
        frame = next(s for m, s in segs if m == 0xC0)
        scan_first = (segs[0][1] + b"".join(s for m, s in segs if m not in (0xD8, 0xC0, 0xDA, 0xD9)) +
                      next(s for m, s in segs if m == 0xDA) + frame + b"\xff\xd9")
        two_frames = b"".join(s + s if m == 0xC0 else s for m, s in segs)
        for name, data in (("sos-before-sof", scan_first), ("two-sof", two_frames),
                           ("no-dqt", no_dqt), ("dqt-wrong-id", wrong_id), ("dqt-after-scan", after_scan),
                           ("dqt1-after-first-scan", late_table1), ("ac-category-15", single_coefficient_jpeg(1, 16384)),
                           ("idct-513", single_coefficient_jpeg(0, 4104))):
            bad.append(json.dumps({"image": base64.b64encode(data).decode(), "out": f"{tmp}/bad-{name}"}))
        raw = b"\0\x10\x20\x30"   # one RGB scanline, including its filter byte
        stream = zlib.compress(raw)
        corrupt_checksum = stream[:-1] + bytes([stream[-1] ^ 1])
        dictionary = zlib.compressobj(zdict=b"dictionary")
        malformed = [corrupt_checksum, stream[:-2], zlib.compress(raw[:-1]),
                     zlib.compress(raw + b"\0"), stream + b"trailing", stream + stream,
                     dictionary.compress(raw) + dictionary.flush()]
        bad.extend(json.dumps({"image": png_stream(s), "out": f"{tmp}/badpng{i}"})
                   for i, s in enumerate(malformed))
        png = base64.b64decode(png_stream(stream))
        for i, offset in enumerate((29, 41 + len(stream), len(png) - 4)):   # IHDR, IDAT, IEND CRCs
            corrupt = bytearray(png)
            corrupt[offset] ^= 1
            bad.append(json.dumps({"image": base64.b64encode(corrupt).decode(), "out": f"{tmp}/badcrc{i}"}))
        for line, got in zip(bad, run_tool(bad)):
            if not got.startswith("ERR"):
                print(f"expected an error for {line[:60]}: {got}")
                failures += 1
        # the crafted JPEGs' unmodified sources still decode, and PIL reads the unusual sampling layout
        Image.open(io.BytesIO(base64.b64decode(LUMA_UNDER_CHROMA_JPEG))).load()
        print("errors: bad base64, truncated PNG, aspect ratio, 16-bit PNG, a non-base64 data URL, a progressive scan "
              "with Se = 255, a luma-under-chroma JPEG, fill before stuffed zeros, JPEGs cut before their first scan, a scan before the frame header, two frame headers, undefined quantization tables, IDCT output beyond [-512, 511], seven malformed zlib "
              "streams and three corrupt CRCs rejected"
              if not failures else "errors: see above")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
