use crate::types::SystemRam;
use std::fs;

pub fn probe_system_ram() -> SystemRam {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0u64;
            let mut avail_kb = 0u64;
            let mut free_kb = 0u64;

            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    total_kb = parse_meminfo_kb(rest);
                } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    avail_kb = parse_meminfo_kb(rest);
                } else if let Some(rest) = line.strip_prefix("MemFree:") {
                    free_kb = parse_meminfo_kb(rest);
                }
            }

            if total_kb > 0 {
                return SystemRam {
                    total_bytes: total_kb.saturating_mul(1024),
                    available_bytes: if avail_kb > 0 {
                        avail_kb.saturating_mul(1024)
                    } else {
                        free_kb.saturating_mul(1024)
                    },
                    free_bytes: free_kb.saturating_mul(1024),
                };
            }
        }
    }

    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages > 0 && page_size > 0 {
        let total = (pages as u64).saturating_mul(page_size as u64);
        SystemRam {
            total_bytes: total,
            available_bytes: total / 2,
            free_bytes: total / 4,
        }
    } else {
        SystemRam {
            total_bytes: 8 * 1024 * 1024 * 1024,
            available_bytes: 4 * 1024 * 1024 * 1024,
            free_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

fn parse_meminfo_kb(s: &str) -> u64 {
    let part = s.split_whitespace().next().unwrap_or("0");
    part.parse().unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_meminfo_kb() {
        assert_eq!(parse_meminfo_kb("  16334580 kB"), 16334580);
        assert_eq!(parse_meminfo_kb("0 kB"), 0);
    }

    #[test]
    fn test_probe_system_ram_nonzero() {
        let ram = probe_system_ram();
        assert!(ram.total_bytes > 0);
        assert!(ram.available_bytes > 0);
    }
}
