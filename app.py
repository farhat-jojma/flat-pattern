from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
import ezdxf
import io
import math
import os
import traceback
import sys

# --- Import shape generators ---
try:
    from shapes import (
        generate_cone,
        generate_frustum_cone,
        generate_frustum_cone_triangulation,
        generate_pyramid,
        generate_rectangle_to_rectangle,
        generate_flange,
        generate_truncated_cylinder,
        generate_elbow  # ✅ new
    )
except Exception as e:
    print("Error importing shapes:", e)
    traceback.print_exc()

sys.stdout.flush()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Flat Pattern API running"})

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        "favicon.ico",
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/generate_dxf", methods=["POST"])
def generate_dxf():
    try:
        data = request.get_json()
        shape = data.get("shape")
        params = data.get("params", {})

        if not shape:
            return jsonify({"error": "Shape not provided"}), 400

        # Create DXF document
        doc = ezdxf.new()
        msp = doc.modelspace()

        # ---------------------- SHAPES ----------------------

        if shape == "cone":
            pts = generate_cone(
                float(params["diameter"]),
                float(params["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "frustum_cone":
            pts = generate_frustum_cone(
                float(params["diameter1"]),
                float(params["diameter2"]),
                float(params["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "frustum_cone_triangulation":
            pts = generate_frustum_cone_triangulation(
                float(params["diameter1"]),
                float(params["diameter2"]),
                float(params["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "pyramid":
            AA = float(params["AA"])
            AB = float(params["AB"])
            H = float(params["H"])
            faces = generate_pyramid(AA, AB, H)
            for face in faces:
                msp.add_lwpolyline(face, close=True)

        elif shape == "rectangle_to_rectangle":
            pts = generate_rectangle_to_rectangle(
                float(params["w1"]),
                float(params["h1"]),
                float(params["w2"]),
                float(params["h2"]),
                float(params["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "flange":
            outer_d = float(params["outer_d"])
            inner_d = float(params["inner_d"])
            holes = int(params["holes"])
            hole_d = float(params["hole_d"])

            pts = generate_flange(outer_d, inner_d, holes, hole_d)
            msp.add_lwpolyline(pts, close=True)

            msp.add_circle((0, 0), inner_d / 2)
            for i in range(holes):
                theta = 2 * math.pi * i / holes
                x = (outer_d / 2 - hole_d) * math.cos(theta)
                y = (outer_d / 2 - hole_d) * math.sin(theta)
                msp.add_circle((x, y), hole_d / 2)

        elif shape == "truncated_cylinder":
            pts = generate_truncated_cylinder(
                float(params["diameter"]),
                float(params["height"]),
                float(params["angle"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "elbow":
            R = float(params["R"])
            alpha = float(params["alpha"])
            D = float(params["D"])
            N = int(params["N"])
            n = int(params["n"])

            result = generate_elbow(R, alpha, D, N, n)

            msp.add_lwpolyline(result["rect"], close=True)
            msp.add_lwpolyline(result["A"])
            msp.add_lwpolyline(result["B"])

        # ✅ NEW: Pipe Elbow (multi-section developed)
        elif shape == "elbow":
            R = float(params["R"])
            alpha = float(params["alpha"])
            D = float(params["D"])
            N = int(params["N"])
            n = int(params["n"])

            result = generate_elbow(R, alpha, D, N, n)

            # Add the 2 viroles (A & B)
            msp.add_lwpolyline(result["A"], close=True)
            msp.add_lwpolyline(result["B"], close=True)

        else:
            return jsonify({"error": f"Shape '{shape}' not supported"}), 400

        # ---------------------- DXF EXPORT ----------------------

        buffer = io.BytesIO()
        try:
            doc.write(stream=buffer)
        except Exception as e:
            text_buffer = io.StringIO()
            doc.write(text_buffer)
            text_data = text_buffer.getvalue().encode("utf-8")
            buffer = io.BytesIO(text_data)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{shape}.dxf",
            mimetype="application/dxf"
        )

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
