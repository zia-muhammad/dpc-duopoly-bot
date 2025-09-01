def test_import_and_stub():
    import duopoly

    # first call: cold start
    price, state = duopoly.p()
    assert isinstance(price, float)
    assert isinstance(state, dict)

    # state carries our compact info; check a couple of known keys
    for k in ["prices_cap", "demands_cap", "step", "direction"]:
        assert k in state

    # second call: pass a demand like the platform would
    price2, state2 = duopoly.p(observed_demand=10.0, information_dump=state)
    assert isinstance(price2, float)
    assert isinstance(state2, dict)
