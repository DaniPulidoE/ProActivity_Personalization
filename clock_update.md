net start w32time
w32tm /resync /force
w32tm /query /status
w32tm /stripchart /computer:pool.ntp.org /samples:3 /dataonly