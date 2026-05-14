import requests

with open("test.csv", "w") as f:
    f.write("A,B\n1,2\n3,4\n")

with open("test.csv", "rb") as f:
    files = {"file": ("test.csv", f, "text/csv")}
    try:
        r = requests.post("http://localhost:8000/api/upload", files=files)
        print("STATUS:", r.status_code)
        print("JSON:", r.json())
    except Exception as e:
        print("ERROR:", e)
