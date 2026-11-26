# Third-party notices

The [project license](LICENSE) covers project-authored material. Separately
sourced components retain their upstream licenses. The release contains
dependency locks, download definitions, and the task configuration. The
deployment host obtains packages, container components and model files from
their upstream sources.

## Runtime components

- TRL 1.19.0, Apache-2.0.
  [Pinned license](https://github.com/huggingface/trl/blob/39c749ed9ce139c594b4e8540fecd0b3cbcf19c9/LICENSE)
- OpenEnv 0.15.0, BSD-3-Clause, Copyright (c) 2026, Hugging Face, Inc.
  [Pinned license](https://github.com/openenv/OpenEnv/blob/v0.15.0/LICENSE)
- Harbor 0.27.0, Apache-2.0.
  [Pinned license](https://github.com/harbor-framework/harbor/blob/v0.27.0/LICENSE)
- Pi coding agent 1.5.2, MIT.
  [Package metadata](https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.5.2)
- Qwen3.8-27B, Apache-2.0.
  [Pinned model license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/LICENSE)

## Task sources

[configs/tasks.json](configs/tasks.json) describes fixtures based on
python-semver and Humanize. Asset preparation downloads the pinned upstream
revisions and preserves their license files when staging the task workspaces.
The original license texts are recorded below.

- python-semver, BSD-3-Clause.
  [Pinned LICENSE.txt](https://github.com/python-semver/python-semver/blob/ec9348aefd8e9f287f451fcf2e837621a40e3ca4/LICENSE.txt)
- Humanize, MIT.
  [Pinned LICENCE](https://github.com/python-humanize/humanize/blob/2968d44280a68f775db73ddeb2f2fc6c05edac84/LICENCE)

### python-semver license

```text
Copyright (c) 2013, Konstantine Rybnikov
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the {organization} nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Humanize license

```text
Copyright (c) 2010-2020 Jason Moiron and Contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
