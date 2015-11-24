End to End Test
======

*(This should probably be deleted and/or replaced)*

**Notice**: this end to end test is not great. I made it for the original version of the code, which is very spaghettified, and difficult to write good tests for. This test is meant only to double check throughout refactoring, that I haven't severely broken anything. Again, you should probably replace it.

How it works
-----

1. Run all the pre-compiling + compiling stuff
1. Run some short evolutions
1. Compare this recent run against the previous run, and make sure the outputs are the same.
