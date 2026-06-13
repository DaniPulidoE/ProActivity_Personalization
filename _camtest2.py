import time, json
import cv2
cam=None
for i in (0,1,2,3):
    cap=cv2.VideoCapture(i); ok=cap.isOpened() and cap.read()[0]; cap.release()
    if ok: cam=i; break
print(f"[camera] index={cam}")
import carla
client=carla.Client("127.0.0.1",2000); client.set_timeout(10.0); world=client.get_world()
bl=world.get_blueprint_library(); bp=bl.find("vehicle.dodge.charger") if bl.filter("vehicle.dodge.charger") else bl.filter("vehicle.*")[0]
vehicle=None
for sp in world.get_map().get_spawn_points()[:10]:
    vehicle=world.try_spawn_actor(bp,sp)
    if vehicle: break
print(f"[carla] vehicle id={getattr(vehicle,'id',None)}")
from ProVoice.decision_engine import CombinedFusionStrategy, XGBoostLoAStrategy, StateXLSTMLoAStrategy
from ProVoice.provoice_actuator import ProVoiceActuator
from ProVoice.data_collector import DataCollector
fcd=XGBoostLoAStrategy(model_path="trained_models/fcd_levels.pkl", default_function="Adjust seat positioning")
state=StateXLSTMLoAStrategy(model_path="trained_models/state_xlstm.pt", default_function="Adjust seat positioning")
strategy=CombinedFusionStrategy(fcd_strategy=fcd, state_strategy=state, w_fcd=0.7)
act=ProVoiceActuator()
dc=DataCollector(visual=True, physiological=True, context=True, sample_rate=20, decision_engine=strategy,
                 actuator=act, cam_index=(cam if cam is not None else 0),
                 static_context={"functionname":"Adjust seat positioning"}, carla_vehicle=vehicle)
dc._calibration_frames=30
labs=set(); emos=set(); bpms=[]; rrs=[]; face_seen=False; first_face=None
try:
    dc.start(); print("[run] started; up to 180s (face perception + rPPG accumulation)...")
    t0=time.time()
    while time.time()-t0 < 180:
        time.sleep(8)
        d=dc.get_latest_data()
        if not dc.calibrated:
            print(f"  ...calibrating ({int(time.time()-t0)}s)"); continue
        if not d: continue
        lab=d.get("lab") or []
        for l in lab: labs.add(l)
        if d.get("emotion"): emos.add(d.get("emotion"))
        fpresent = "face" in lab
        if fpresent and not face_seen: face_seen=True; first_face=int(time.time()-t0)
        realhr = "bpm" in d  # rPPG sets 'bpm'; random fallback does not
        if realhr: bpms.append(d.get("bpm"))
        if "breaths-per-minute" in d: rrs.append(d.get("breaths-per-minute"))
        print("[live] " + json.dumps({k:d.get(k) for k in
              ("eye_ar","mar","gaze_score","gaze_distracted","blink_count","yawn_count","perclos",
               "drowsiness_alert","emotion","emotion_prob","lab","heart_rate","bpm","speed")}, default=str))
    print("\n===== SUMMARY =====")
    print("face detected:", face_seen, "(first at", first_face, "s)" if face_seen else "")
    print("distraction labels seen:", sorted(labs))
    print("emotions seen:", sorted(emos))
    print("real rPPG HR samples:", len(bpms), ("(bpm range %.0f-%.0f)"%(min(bpms),max(bpms)) if bpms else "(none - needs more frames)"))
    print("real rPPG RR samples:", len(rrs))
    print("final blink_count:", dc.blink_count, "yawn_count:", dc.yawn_count)
    print("calibration thresholds:", json.dumps(getattr(dc,"calibrate",{}), default=str))
finally:
    dc.stop()
    if vehicle: vehicle.destroy(); print("[carla] temp vehicle destroyed")
