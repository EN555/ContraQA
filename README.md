# ContraQA

## Getting Started

```bash
$ conda create -n contraqa 'python<3.13'
$ conda activate contraqa
$ pip install -r requirements.txt -r requirements-dev.txt
```

create an `.env` file for OpenAI key

```bash
echo 'OPENAI_API_KEY=<mykey>' > .env
```

Formatting file with `black`

```bash
$ black --line-length 100 --skip-string-normalization --target-version py312 src/*.py
```
