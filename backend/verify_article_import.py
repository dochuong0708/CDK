from app.main import app

for route in app.routes:
    if getattr(route, "path", None) == "/api/analyze/article":
        print("Article endpoint:", route.path)

print("Application imported successfully.")
