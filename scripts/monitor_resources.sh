#!/bin/bash
# Resource monitor for KSERESNET-RS LLM runs on the DGX MIG partition.
#
# KNOWN LIMITATIONS (documented, not silently ignored):
# 1. Per-MIG SM/compute utilization percentage is NOT exposed by nvidia-smi
#    for MIG instances (confirmed via `nvidia-smi -q`) -- only FB memory
#    usage is available per MIG device. GPU "utilization" in our results
#    therefore means MEMORY utilization, not compute/SM utilization, and
#    must be labeled as such everywhere it is reported.
# 2. This is a shared, multi-tenant machine with no cgroup CPU/RAM limit
#    applied to our process -- "CPU utilization" and "RAM" figures reflect
#    SYSTEM-WIDE usage (all users' processes), not usage attributable only
#    to our job. nproc reports 128 visible threads; this is NOT a dedicated
#    allocation.
# 3. GPU power draw and temperature ARE available system-wide via nvidia-smi
#    but are reported per physical GPU (not per MIG instance) -- recorded
#    for the physical GPU hosting our MIG slice (index 2).
#
# Usage: ./monitor_resources.sh <output_csv> <interval_seconds> [duration_seconds]
# duration_seconds omitted or 0 = run until killed.

OUT="${1:-resource_log.csv}"
INTERVAL="${2:-3}"
DURATION="${3:-0}"

GPU_INDEX=2          # physical GPU hosting our MIG slice (81:00.0)
MIG_DEVICE_INDEX=1   # MIG Device Index within that GPU (matches MIG-80f109f6...)

echo "timestamp,mig_mem_used_mib,mig_mem_total_mib,mig_mem_used_pct,gpu_power_draw_w,gpu_temp_c,sys_cpu_util_pct,sys_ram_used_mib,sys_ram_total_mib,sys_ram_used_pct,sys_load_1min" > "$OUT"

# prime /proc/stat for delta-based CPU% calculation
read -r _ u1 n1 s1 i1 w1 irq1 sirq1 _ < /proc/stat

start_time=$(date +%s)
while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%S")

    mig_block=$(nvidia-smi -q -i "$GPU_INDEX" 2>/dev/null | awk -v idx="$MIG_DEVICE_INDEX" '
        /Index/{cur_idx=$3}
        cur_idx==idx{print}
    ')
    mem_used=$(echo "$mig_block" | grep -m1 "Used" | grep -oE '[0-9]+' | head -1)
    mem_total=$(echo "$mig_block" | grep -m1 "Total" | grep -oE '[0-9]+' | head -1)
    [ -z "$mem_used" ] && mem_used=0
    [ -z "$mem_total" ] && mem_total=40192
    mem_pct=$(awk -v u="$mem_used" -v t="$mem_total" 'BEGIN{if(t>0) printf "%.1f", (u/t)*100; else print "0"}')

    power_temp=$(nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null)
    gpu_power=$(echo "$power_temp" | cut -d',' -f1 | tr -d ' ')
    gpu_temp=$(echo "$power_temp" | cut -d',' -f2 | tr -d ' ')
    [ -z "$gpu_power" ] && gpu_power="NA"
    [ -z "$gpu_temp" ] && gpu_temp="NA"

    # system-wide CPU utilization via /proc/stat delta
    read -r _ u2 n2 s2 i2 w2 irq2 sirq2 _ < /proc/stat
    total1=$((u1+n1+s1+i1+w1+irq1+sirq1))
    total2=$((u2+n2+s2+i2+w2+irq2+sirq2))
    idle1=$i1
    idle2=$i2
    dtotal=$((total2-total1))
    didle=$((idle2-idle1))
    if [ "$dtotal" -gt 0 ]; then
        cpu_pct=$(awk -v dt="$dtotal" -v di="$didle" 'BEGIN{printf "%.1f", (1-(di/dt))*100}')
    else
        cpu_pct="0.0"
    fi
    u1=$u2; n1=$n2; s1=$s2; i1=$i2; w1=$w2; irq1=$irq2; sirq1=$sirq2

    ram_line=$(free -m | awk '/^Mem:/{print $3","$2}')
    ram_used=$(echo "$ram_line" | cut -d',' -f1)
    ram_total=$(echo "$ram_line" | cut -d',' -f2)
    ram_pct=$(awk -v u="$ram_used" -v t="$ram_total" 'BEGIN{if(t>0) printf "%.1f", (u/t)*100; else print "0"}')

    load1=$(cut -d' ' -f1 /proc/loadavg)

    echo "$ts,$mem_used,$mem_total,$mem_pct,$gpu_power,$gpu_temp,$cpu_pct,$ram_used,$ram_total,$ram_pct,$load1" >> "$OUT"

    if [ "$DURATION" -gt 0 ]; then
        now=$(date +%s)
        elapsed=$((now - start_time))
        if [ "$elapsed" -ge "$DURATION" ]; then
            break
        fi
    fi

    sleep "$INTERVAL"
done
