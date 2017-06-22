Changing the order of two lines in these files, changes the runtime by
about a factor 2.

This probably relates to inadverted (ab)use of global objects, and at
what point we bind to which. We don't understand this at the moment,
it will probably become clear later.
