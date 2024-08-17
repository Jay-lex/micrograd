use micrograd::engine::Value;

fn main() {
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

    println!("{:.4}", g.borrow().data); // prints 24.7041, the outcome of this forward pass
    g.backward();
    println!("{:.4}", a.borrow().grad); // print 138.8338, i.e. the numerical value of dg/da
    println!("{:.4}", b.borrow().grad); // print 645.5773, i.e. the numerical value of dg/db
}
