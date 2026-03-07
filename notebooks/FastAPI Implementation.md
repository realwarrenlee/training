# Fastapi Implementation

Today we will be learning how to serve a model using Fastapi.

> Learn basic Fastapi functions\
> Attach trained STT model\
> Produce your very own model API

*Step 1. Import Dependencies*

Make a main.py

- Load essential libraries:

        pip install fastapi==0.133.1 uvicorn==0.29.0 python-multipart==0.0.9 torch==2.6.0 transformers==4.48.2 librosa==0.12.1 numpy==1.28.2 soundfile==0.12.1 resampy==0.4.3

API framework (FastAPI)
Audio processing (librosa, numpy, soundfile, resampy)
Speech recognition (transformers)
Hardware acceleration (torch)
Web server (uvicorn)
Streaming parser (python-multipart)

*Step 2. Init API app*

- Use the below format to establish important information about your API, e.g.

        from fastapi import FastAPI

        app = FastAPI(
            title="Whisper STT API",
            description="Speech-to-Text API using Whisper models",
            version="0.1.0"
        )

        @app.get("/")
        async def root():
            return {
                "title": app.title,
                "description": app.description,
                "version": app.version,
            }

- Then, try to run using uvicorn

        uvicorn notebooks.test:main --reload --port 8000

- Check http://127.0.0.1:8000/ and http://127.0.0.1:8000/docs

Play around a bit to understand.

*Step 3. Check Health*

- Add a /health endpoint

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if ("model_loaded" in globals() and model_loaded) else "loading"
            }

- Play around and understand.

- For now you should see an unhealthy model, because you havent loaded your trained model yet.

Let's proceed to the next step to do that.

*Step 4. Load Model*

- Using @app.on_event("startup") and an async def, load the trained model at startup.

- Use your normal code and try to go off what you have already seen.

- Remember to set any model components, e.g. model, processor as global variables.

- Use try statements to prevent unexpected crashes.

*Step 5. Serve Model*

- Create a transcription endpoint.

- You may use the following to read an API call.

        content = await file.read()

- Then, run your model inference code as per normal.

*Step 6. Test API*

- You may use Python requests library tosend a request, e.g.

        import requests

        url = "http://127.0.0.1:8000/transcribe"
        audio_path = "path/to/your/audio.wav"

        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path, audio_file, "audio/wav")}
            response = requests.post(url, files=files)

        print(response.json())
        # Output: {"text": "Transcribed speech content here"}