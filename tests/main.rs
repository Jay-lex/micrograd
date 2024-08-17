/*
----------------------------------------------------------------------------------
This is a test to make sure that the tests in the examples pass.
----------------------------------------------------------------------------------
*/

use micrograd::engine::Value;

#[test]
pub fn test_usage() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);

    let mut c = &a + &b;
    let mut d = &a * &b + b.pow(3.0);

    c += &c + 1.0;
    c += 1.0 + &c + (-&a);
    d += &d * 2.0 + (&b + &a).relu();
    d += 3.0 * d + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(2.0);
    let mut g = &f / 2.0;
    g += 10.0 / &f;

    assert_eq!(format!("{:.4}", g.borrow().data), "24.7041");
    g.backward();
    assert_eq!(format!("{:.4}", a.borrow().grad), "138.8338");
    assert_eq!(format!("{:.4}", b.borrow().grad), "645.5773");
}
