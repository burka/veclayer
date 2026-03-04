//! Three-tier capability model for VecLayer authorization.

use serde::{Deserialize, Serialize};

/// Authorization capability level.
///
/// Read < Write < Admin — each higher level includes all lower capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Read,
    Write,
    Admin,
}

impl Capability {
    /// Check if this capability is sufficient for the required level.
    pub fn permits(&self, required: Capability) -> bool {
        *self >= required
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

impl std::str::FromStr for Capability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "admin" => Ok(Self::Admin),
            other => Err(format!("unknown capability: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_ordering() {
        assert!(Capability::Read < Capability::Write);
        assert!(Capability::Write < Capability::Admin);
        assert!(Capability::Read < Capability::Admin);
    }

    #[test]
    fn test_permits() {
        assert!(Capability::Admin.permits(Capability::Write));
        assert!(Capability::Admin.permits(Capability::Read));
        assert!(Capability::Admin.permits(Capability::Admin));
        assert!(Capability::Write.permits(Capability::Read));
        assert!(Capability::Write.permits(Capability::Write));
        assert!(!Capability::Write.permits(Capability::Admin));
        assert!(Capability::Read.permits(Capability::Read));
        assert!(!Capability::Read.permits(Capability::Write));
        assert!(!Capability::Read.permits(Capability::Admin));
    }

    #[test]
    fn test_serde_roundtrip() {
        for cap in [Capability::Read, Capability::Write, Capability::Admin] {
            let json = serde_json::to_string(&cap).expect("serialize");
            let recovered: Capability = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(cap, recovered);
        }
    }

    #[test]
    fn test_serde_lowercase_strings() {
        assert_eq!(
            serde_json::to_string(&Capability::Read).unwrap(),
            "\"read\""
        );
        assert_eq!(
            serde_json::to_string(&Capability::Write).unwrap(),
            "\"write\""
        );
        assert_eq!(
            serde_json::to_string(&Capability::Admin).unwrap(),
            "\"admin\""
        );
    }

    #[test]
    fn test_from_str() {
        use std::str::FromStr;
        assert_eq!(Capability::from_str("read").unwrap(), Capability::Read);
        assert_eq!(Capability::from_str("write").unwrap(), Capability::Write);
        assert_eq!(Capability::from_str("admin").unwrap(), Capability::Admin);
        // Case-insensitive
        assert_eq!(Capability::from_str("READ").unwrap(), Capability::Read);
        assert_eq!(Capability::from_str("Admin").unwrap(), Capability::Admin);
        // Unknown value
        assert!(Capability::from_str("unknown").is_err());
        assert!(Capability::from_str("superuser").is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(Capability::Read.to_string(), "read");
        assert_eq!(Capability::Write.to_string(), "write");
        assert_eq!(Capability::Admin.to_string(), "admin");
    }
}
