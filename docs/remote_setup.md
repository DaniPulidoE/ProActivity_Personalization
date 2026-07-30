# Two-machine setup over a direct Ethernet link

The study runs across two machines: **CARLA machine** (simulator, NPC traffic,
Drive with the LoA popups, both bridges) and **ProVoice machine** (perception,
models, `raw_data.jsonl`). They are joined by a dedicated Ethernet cable; both
keep Wi-Fi for the internet.

| | CARLA machine | ProVoice machine |
|---|---|---|
| Link address | `192.168.50.1` | `192.168.50.2` |
| Adapter name | `Ethernet 2` | `Ethernet 8` (ASIX USB dongle) |
| Listens on | TCP 8080, 8081 | nothing |

Adapter names are per-machine and can change — verify with `Get-NetAdapter`
rather than trusting this table. The addresses are pinned in
`start_experiment.py` as `CARLA_MACHINE_IP`; change it there if the rig is
re-addressed.

## What actually crosses the cable

Worth knowing precisely, because it is what the ethics paperwork has to match:

| Direction | Payload |
|---|---|
| CARLA → ProVoice, ~20 Hz | speed, throttle, brake, steer, gear, weather, junction, traffic light, indicators |
| CARLA → ProVoice, once | `session_id`, `participantid`, `status_url` |
| ProVoice → CARLA, twice | `collection_started` / `provoice_ended`, with `session_id` and `participantid` |

**No participant data crosses the link.** Camera frames and everything derived
from them — PERCLOS, gaze, EAR/MAR, HR/RR, emotion, distraction labels — are
computed and written on the ProVoice machine and stay there. The dashboard that
renders the webcam overlay binds `127.0.0.1`. The only personal datum on the
wire is the pseudonymous participant ID.

Neither bridge authenticates, so the network scoping below *is* the access
control: bind to the link only, and firewall to the one peer.

---

# One-time setup

## 1. Physical

Cable directly between the two NICs. Any modern adapter auto-negotiates; a
crossover cable is not needed.

## 2. Identify the adapter (both machines)

```powershell
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed -AutoSize
```

Take the **physical** NIC that reads `Up` — a real description (Intel, Realtek,
ASIX), not `VMware`, `VirtualBox Host-Only`, `vEthernet (...)`, `TeamViewer VPN`,
`AnyConnect` or `PANGP`. A USB dongle with nothing plugged in reads
`Disconnected`, so `Up` is the confirmation that both ends are live.

> **Stop if that adapter is a campus wall port.** The rest of this puts a static
> address and listening services on the university network.

## 3. Static addresses (admin PowerShell, both machines)

```powershell
# CARLA machine
Set-NetIPInterface       -InterfaceAlias "Ethernet 2" -Dhcp Disabled
New-NetIPAddress         -InterfaceAlias "Ethernet 2" -IPAddress 192.168.50.1 -PrefixLength 24
Set-NetConnectionProfile -InterfaceAlias "Ethernet 2" -NetworkCategory Private

# ProVoice machine
Set-NetIPInterface       -InterfaceAlias "Ethernet 8" -Dhcp Disabled
New-NetIPAddress         -InterfaceAlias "Ethernet 8" -IPAddress 192.168.50.2 -PrefixLength 24
Set-NetConnectionProfile -InterfaceAlias "Ethernet 8" -NetworkCategory Private
```

**No default gateway and no DNS.** A gateway here would try to route internet
traffic down the cable and break Wi-Fi.

Without this the link self-assigns APIPA (`169.254.x.x`), which works but is slow
to negotiate and changes between reboots. `Private` matters because Windows
classifies a new link as `Public`, where inbound connections are blocked — that
is what makes a POST **time out** rather than be refused.

### Fixing a mistake

```powershell
Get-NetIPAddress    -InterfaceAlias "Ethernet 2" -AddressFamily IPv4
Remove-NetIPAddress -InterfaceAlias "Ethernet 2" -AddressFamily IPv4 -Confirm:$false
New-NetIPAddress    -InterfaceAlias "Ethernet 2" -IPAddress 192.168.50.1 -PrefixLength 24
```

If a default gateway was set by accident, remove the route it left behind:

```powershell
Get-NetRoute    -InterfaceAlias "Ethernet 2" -AddressFamily IPv4
Remove-NetRoute -InterfaceAlias "Ethernet 2" -DestinationPrefix 0.0.0.0/0 -Confirm:$false
```

## 4. Firewall — CARLA machine only

```powershell
New-NetFirewallRule -DisplayName "ProActivity bridges" -Direction Inbound `
  -Protocol TCP -LocalPort 8080,8081 -Action Allow `
  -InterfaceAlias "Ethernet 2" -RemoteAddress 192.168.50.2

New-NetFirewallRule -DisplayName "ProActivity ping" -Direction Inbound `
  -Protocol ICMPv4 -IcmpType 8 -Action Allow `
  -InterfaceAlias "Ethernet 2" -RemoteAddress 192.168.50.2
```

Scoped to the interface **and** the one peer, so the rules do not also open
these ports on Wi-Fi, which Windows may classify `Private` too. The ProVoice
machine needs no rule — it only connects outbound.

## 5. Smoke test

On the CARLA machine, with CARLA running:

```
uv run python start_experiment.py --experiment-data-collection-carla-remote --participantid TEST
```

From the ProVoice machine:

```powershell
ping 192.168.50.1
curl.exe http://192.168.50.1:8080/health
curl.exe http://192.168.50.1:8080/session    # session_id, participantid=TEST, status_url
curl.exe http://192.168.50.1:8081/health
```

A hang means firewall or wrong address; an instant refusal means the server is
not running. Ctrl-C afterwards and discard anything written for `TEST`.

---

# Per session

```
Phase 0  teaching the LoA control
  CARLA:     uv run python start_experiment.py --experiment-popup

Phase 1  driver adaptation
  CARLA:     uv run python start_experiment.py --experiment-adaptation

Phase 2  calibration (180 s baseline)
  CARLA:     uv run python start_experiment.py --experiment-calibration-carla-remote --participantid 007
  PROVOICE:  uv run python start_experiment.py --experiment-calibration-provoice-remote

Phase 3  data collection
  CARLA:     uv run python start_experiment.py --experiment-data-collection-carla-remote --participantid 007
  PROVOICE:  uv run python start_experiment.py --experiment-data-collection-provoice-remote
```

No addresses on either side: both machines run the same `start_experiment.py`
and take the link address from `CARLA_MACHINE_IP`. Each ProVoice command prints
where it got it —

```
[PRESET] bridge at http://192.168.50.1:8080 (from the pinned rig address)
```

— which is worth a glance, because the pinned default is only right while both
machines are on the same revision of the file. Off the rig, pass an address
(`--experiment-data-collection-provoice-remote 10.0.0.7`) or set
`PV_BRIDGE_URL`; both take precedence over the constant.

`--participantid` is typed **once**, on the CARLA machine. The ProVoice machine
reads it and the session id from `192.168.50.1:8080/session`, and refuses to
start if a value given there disagrees.

Start the CARLA side first each time; ProVoice waits ~10 s for the ids either
way. Before anyone drives, check the ProVoice terminal for:

```
[bridge] adopting session_id=... / participantid=... from the bridge.
[status] reverse bridge reachable at http://192.168.50.1:8081
```

The second line is the confirmation that Drive will be told when recording
starts, and when the session ends. Without it, the LoA windows fall back to a
180 s timeout and the drive has to be stopped by hand.

## After each participant

The halves log to their own machines — `raw_data.jsonl` on the ProVoice machine,
`user_loa_labels.csv` on the CARLA machine. Copy them together before running
`scripts/build_loa_dataset.py`; they join on the shared session id.

---

# Reconnecting the cable

Unplugging and replugging does **not** require redoing the setup. Static
addresses and firewall rules are stored per adapter and survive replug and
reboot. Two things can bite, so run this on both machines before a session:

```powershell
Get-NetAdapter          -InterfaceAlias "Ethernet 2"   # Up?
Get-NetIPAddress        -InterfaceAlias "Ethernet 2" -AddressFamily IPv4   # still 192.168.50.x?
Get-NetConnectionProfile -InterfaceAlias "Ethernet 2"  # still Private?
```

- **The profile can revert to `Public`.** Windows re-identifies the link when it
  comes back, and an unidentified network can land in the wrong category again.
  Symptom: everything times out. Fix: re-run the one `Set-NetConnectionProfile`
  line. Nothing else needs repeating.
- **A USB dongle in a different port may enumerate as a NEW adapter** (say
  `Ethernet 9`) with no address and no matching firewall rule, since the rules
  are bound to the interface name. Use the same USB port every time; if the name
  did change, redo steps 3–4 for the new one.

A `169.254.x.x` address on the adapter means the static configuration is gone —
redo step 3.

# Teardown at the end of the study

```powershell
Remove-NetFirewallRule -DisplayName "ProActivity bridges"
Remove-NetFirewallRule -DisplayName "ProActivity ping"
```

# Troubleshooting

| Symptom | Cause |
|---|---|
| POST **times out** | Firewall (port not open, or profile went `Public`), or wrong address. A dropped SYN, not a closed port. |
| Connection **refused** immediately | Address is right and reachable; the server is not running. Check the launcher started `VEHICLE_SERVER` / `STATUS_SERVER`. |
| `[status] ... did not answer /health` at ProVoice startup | Reverse channel down. The run still records everything, but Drive will not learn when it starts or ends. |
| `/session` shows `null` ids | The CARLA launcher was started without `--participantid`, or an old bridge is still running from a previous phase. |
| Works, then stops mid-session | Check whether a VPN client (GlobalProtect, AnyConnect) connected and pushed a tunnel-all route. |
| ProVoice exits at once with `[FATAL] ... participant id` | Calibration with no participant id — the bridge published none. |
