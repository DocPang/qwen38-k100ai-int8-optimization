# Third-party notices

This repository contains patches/adaptations for third-party projects. It does
**not** redistribute the SourceFind base Docker image or model weights.

## SGLang

Modified SGLang files retain their upstream Apache-2.0 notices and remain
subject to the upstream SGLang license.

## DFlash / Z Lab

Parts of the DFlash2 integration are derived from or informed by `z-lab/dflash`.
The upstream project is distributed under the MIT License:

MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Models and vendor image

- `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`: downloaded separately and
  subject to its upstream model/repository terms.
- `z-lab/Qwen3.8-27B-DFlash2`: downloaded separately and subject to its upstream
  model/repository terms.
- SourceFind K100AI SGLang Docker image: referenced as a `FROM` base only and
  not included in this repository. Users must obtain it through the vendor's
  normal distribution channel and comply with its terms.
