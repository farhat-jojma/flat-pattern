from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ezdxf
import io
import math
import os
import traceback
import sys
import base64

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
        generate_bend,
        generate_circle_to_rectangle,
        generate_offset_cone,
        generate_sphere,
        generate_rectangle_to_circle,
        generate_rectangle_to_circle_ecc,
        generate_frustum_ecc_angle,
        generate_frustum_ecc_paral,
        generate_auger,
        generate_breeches_full,
        generate_offset_tee,
        generate_tee_oblique,
        generate_tee_eccentric,
        generate_tee_on_bend,
        generate_tee_on_cone,
        generate_pants,
        generate_pants2,
        generate_pants_ecc
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
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
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
        response_data = {}

        # ---------------------- SHAPES ----------------------

        if shape == "cone":
            res = generate_cone(float(params["diameter"]), float(params["height"]))
            msp.add_lwpolyline(res["points"], close=True)
            response_data = res.get("data", {})

        elif shape == "frustum_cone":
            d1 = float(params.get("D1") or params.get("diameter1"))
            d2 = float(params.get("D2") or params.get("diameter2"))

            # Determine calculation mode
            mode = params.get("mode", "H").upper()
            value = float(params.get("H") if mode == "H" else params.get("beta"))

            res = generate_frustum_cone(d1, d2, value, mode)
            msp.add_lwpolyline(res["points"], close=True)
            response_data = res.get("data", {})

        elif shape == "frustum_cone_triangulation":
            d1 = float(params.get("D1") or params.get("diameter1"))
            d2 = float(params.get("D2") or params.get("diameter2"))
            H = float(params.get("H") or params.get("height"))
            n = int(params.get("n", 12))

            res = generate_frustum_cone_triangulation(d1, d2, H, "H", n)
            msp.add_lwpolyline(res["points"], close=True)
            for line in res["generators"]:
                msp.add_line(line[0], line[1])
            response_data = res.get("data", {})

        elif shape == "pyramid":
            AA, AB, H = float(params["AA"]), float(params["AB"]), float(params["H"])
            res = generate_pyramid(AA, AB, H)
            for face in res["faces"]:
                msp.add_lwpolyline(face, close=True)
            response_data = res["data"]

        elif shape == "rectangle_to_rectangle":
            res = generate_rectangle_to_rectangle(
                float(params["ab"]), float(params["bc"]), float(params["H"]),
                float(params["AB"]), float(params["BC"])
            )
            for face in res["faces"]:
                msp.add_lwpolyline(face, close=True)
            response_data = res["data"]

        elif shape == "flange":
            res = generate_flange(
                float(params["D1"]), float(params["D2"]),
                float(params["D3"]), float(params["D4"]),
                int(params["N1"]), float(params["d1"]),
                int(params["N2"]), float(params["d2"]),
            )
            for etype, center, radius in res["entities"]:
                x, y = center
                steps = 90
                pts = [(x + radius * math.cos(2 * math.pi * i / steps),
                        y + radius * math.sin(2 * math.pi * i / steps))
                       for i in range(steps + 1)]
                msp.add_lwpolyline(pts, close=True)
            response_data = res["data"]

        elif shape == "truncated_cylinder":
            out = generate_truncated_cylinder(
                float(params["diameter"]),
                float(params["height"]),
                float(params["angle"]),
                int(params["n"])
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "bend":
            out = generate_bend(
                float(params["R"]),
                float(params["alpha"]),
                float(params["D"]),
                int(params["N"]),
                int(params["n"]),
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "circle_to_rectangle":
            out = generate_circle_to_rectangle(
                float(params["D"]),
                float(params["H"]),
                float(params["A"]),
                float(params["B"]),
                int(params["n"])
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "offset_cone":
            out = generate_offset_cone(
                float(params["D"]),
                float(params["H"]),
                float(params["X"]),
                int(params["n"])
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "sphere":
            out = generate_sphere(
                float(params["D"]),
                int(params["N"]),
                int(params["n"])
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "rectangle_to_circle":
            out = generate_rectangle_to_circle(
                float(params["D"]),
                float(params["H"]),
                float(params["A"]),
                float(params["B"]),
                int(params["n"])
            )
            response_data = out.get("calc", out.get("data", out))

        elif shape == "rectangle_to_circle_ecc":
            out = generate_rectangle_to_circle_ecc(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "frustum_ecc_angle":
            out = generate_frustum_ecc_angle(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "frustum_ecc_paral":
            out = generate_frustum_ecc_paral(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "auger":
            out = generate_auger(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "breeches_full":
            out = generate_breeches_full(params, msp=msp, layer="CUT")
            response_data = out["calc"]
        
        elif shape == "offset_tee":
            out = generate_offset_tee(params, msp=msp, layer="CUT")
            response_data = out["calc"]
            
        elif shape == "tee_oblique":
            out = generate_tee_oblique(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "tee_eccentric":
            out = generate_tee_eccentric(params, msp=msp, layer="CUT")
            response_data = out["calc"]
        
        elif shape == "tee_on_bend":
            out = generate_tee_on_bend(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "tee_on_cone":
            out = generate_tee_on_cone(params, msp=msp, layer="CUT")
            response_data = out["calc"]
            
        elif shape == "pants":
            out = generate_pants(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "pants2":
            out = generate_pants2(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        elif shape == "pants_ecc":
            out = generate_pants_ecc(params, msp=msp, layer="CUT")
            response_data = out["calc"]

        else:
            return jsonify({"error": f"Shape '{shape}' not supported"}), 400

                # ---------------------- DXF EXPORT ----------------------
        try:
            import io
            from io import StringIO

            buf = StringIO()
            doc.write(buf)  # write DXF to memory as text
            dxf_str = buf.getvalue().encode("utf-8")
            dxf_base64 = base64.b64encode(dxf_str).decode("utf-8")
        except Exception as e:
            print("Error writing DXF:", e)
            traceback.print_exc()
            return jsonify({"error": f"DXF write failed: {str(e)}"}), 500

        # ✅ Unified JSON response
        return jsonify({
            "shape": shape,
            "data": response_data,
            "dxf_base64": dxf_base64
        })

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
