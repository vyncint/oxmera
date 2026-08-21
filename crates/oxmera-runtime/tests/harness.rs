//! Harness placeholder for the runtime seam.
//!
//! The real dispatch specs run once a backend exists (rung A4). Ignored
//! until then.

use oxmera_core::{Device, Error};
use oxmera_runtime::backend_for;

#[test]
#[ignore = "unignore after exercise A4 is solved; dispatch needs a registered backend to test against"]
fn unregistered_device_is_a_typed_error_not_a_panic() {
    let result = backend_for(Device::Cuda { index: 0 });
    assert!(matches!(
        result.map(|_| ()),
        Err(Error::BackendUnavailable { .. })
    ));
}
