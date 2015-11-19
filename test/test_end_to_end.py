'''

End to end test

Maybe a temporary test, it's mostly for making sure that the results stay the same throughout my refactoring

fixtures
* need a yaml
* config.json

rm -r build && python setup.py build_ext --inplace

python pyanimats.py \
"./test/end to end/raw_results/0.0.22/initial_tests/seed-152" \
"./test/end to end/fixtures/tasks/3-4-6-5.yml" \
--num-gen=100 \
--num-samples=10 \
--seed=152 \
--config="./test/end to end/fixtures/tasks/config.json"


'''

