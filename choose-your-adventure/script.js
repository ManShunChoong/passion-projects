// const player = {
//     visitedLocationIds: ["barn0", "gardenA", "gardenB0", "gardenB"],
//     metCharacterIds: ["pig"]
// }
const player = {
    visitedLocationIds: [],
    metCharacterIds: [],
    chapter: 0
}
const locationNodes = [
    {
        id: "barn0",
        text: [
            "You open your eyes.",
            "You are lying down on huge bundles of hay in a dark barn. There is a big gate on your right."
        ],
        image: "barn.png",
        options: [
            {
                text: "Go through the gate.",
                nextId: "gardenA"
            },
            {
                text: "Go back to sleep.",
                nextId: "barn0"
            }
        ]
    },
    {
        id: "barn",
        status: "Barn",
        text: [
            "Huge bundles of hay lying all around the dark barn.",
            "There is only this one big gate."
        ],
        image: "barn.png",
        options: [
            {
                text: "Go out from the barn.",
                nextId: "gardenA"
            },
            {
                text: "Notice something moving in hay.",
                nextId: 1,
                nextKey: "sleepingPig",
                requiredState: (player) => player.metCharacterIds.includes("pig") && player.metCharacterIds.includes("luna") && !player.wokePig
            }
        ]
    },
    {
        id: "gardenA",
        status: "Garden",
        text: [
            "You are looking at a beautiful garden next to the barn. It has flowers and trees everywhere.",
            "A perfect place to walk around. You can also hear little water splashing sound from here."
        ],
        image: "garden.png",
        options: [
            {
                text: "Walk around the garden.",
                nextId: "gardenB0",
                requiredState: (player) => !player.visitedLocationIds.includes("gardenB0")
            },
            {
                text: "Walk around the garden.",
                nextId: "gardenB",
                requiredState: (player) => player.visitedLocationIds.includes("gardenB0")
            },
            {
                text: "Approach some cats.",
                nextId: 1,
                nextKey: "lunaText",
                requiredState: (player) => player.metCharacterIds.includes("pig") && !player.metCharacterIds.includes("luna")
            },
            {
                text: "Approach some cats.",
                nextId: 1,
                nextKey: "catText",
                requiredState: (player) => player.metCharacterIds.includes("luna")
            },
            {
                text: "Go into the barn.",
                nextId: "barn"
            },
            {
                text: "Walk towards the water sound.",
                nextId: "pond",
                requiredState: (player) => !player.visitedLocationIds.includes("pond")
            },
            {
                text: "Walk towards the pond.",
                nextId: "pond",
                requiredState: (player) => player.visitedLocationIds.includes("pond")
            }
        ]
    },
    {
        id: "gardenB0",
        status: "Garden",
        text: [
            "You walk around the garden.",
            "Beautiful flowers... But bugs?!\n",
            "You feel a touch on your shoulder."
        ],
        image: "garden.png",
        options: [
            {
                text: "Turn around.",
                nextId: 1,
                nextKey: "pigText",
                requiredState: (player) => !player.metCharacterIds.includes("pig"),
                setState: (player) => {
                    if (!player.metCharacterIds.includes("pig")) {
                        player.metCharacterIds.push("pig")
                    }
                }
            }
        ]
    },
    {
        id: "gardenB",
        status: "Garden",
        text: [
            "You walk around the garden.",
            "Beautiful flowers... But bugs? No..."
        ],
        image: "garden.png",
        options: [
            {
                text: "Walk around the garden.",
                nextId: "gardenA",
            }
        ]
    },
    {
        id: "pond",
        status: "Pond",
        text: [
            "You walk towards the water splashing sound.",
            "It is a beautiful pond. You can see a big fish swimming in it."
        ],
        image: "pond.png",
        options: [
            {
                text: "Walk to the garden.",
                nextId: "gardenA"
            },
            {
                text: "Walk to the field.",
                nextId: "fieldA",
                requiredState: (player) => player.metCharacterIds.includes("luna") && player.group
            },
            {
                text: "Look for yummy food.",
                nextId: 1,
                nextKey: "foodText",
                requiredState: (player) => player.visitedLocationIds.includes("gardenB0")
            },
            {
                text: "Find the food stall old man.",
                nextId: 1,
                nextKey: "oldManText",
                requiredState: (player) => player.metCharacterIds.includes("oldMan")
            }
        ]
    },
    {
        id: "foodStall",
        status: "Food Stall",
        text: [
            "There are quite a number of people around.",
            "They are all eating food."
        ],
        image: "foodStall.png",
        options: [
            {
                text: "Walk around the pond.",
                nextId: "pond"
            },
            {
                text: "Talk to food stall old man.",
                nextId: 1,
                nextKey: "oldManText"
            },
            {
                text: "Talk to the group nearest to you.",
                nextId: 1,
                nextKey: "adventurerText",
                setState: (player) => {
                    if (!player.metCharacterIds.includes("adventurer")) {
                        player.metCharacterIds.push("adventurer")
                    }
                },
                requiredState: (player) => !player.metCharacterIds.includes("adventurer")
            },
            {
                text: "Talk to the adventurer group.",
                nextId: 1,
                nextKey: "adventurerText",
                requiredState: (player) => player.metCharacterIds.includes("adventurer") && !player.group
            },
            {
                text: "Talk to the black clothing group.",
                nextId: 1,
                nextKey: "banditText",
                setState: (player) => {
                    if (!player.metCharacterIds.includes("bandit")) {
                        player.metCharacterIds.push("bandit")
                    }
                },
                requiredState: (player) => !player.metCharacterIds.includes("bandit")
            },
            {
                text: "Talk to the bandit group.",
                nextId: 1,
                nextKey: "banditText",
                requiredState: (player) => player.metCharacterIds.includes("bandit") && !player.group
            }
        ]
    },
    {
        id: "fieldA",
        status: "Field",
        text: [
            "You swim across the pond and see a field of...",
            "Wheat? Corn? You can't see clearly.\n",
            "While people are walking around the pond to reach the field."
        ],
        image: "field.png",
        options: [
            {
                text: "Walk to the wooden houses.",
                nextId: "woodenHouse0",
                requiredState: (player) => !player.visitedLocationIds.includes("woodenHouse0")
            },
            {
                text: "Walk to the wooden houses.",
                nextId: "woodenHouse",
                requiredState: (player) => player.visitedLocationIds.includes("woodenHouse0")
            }
        ]
    },
    {
        id: "woodenHouse0",
        status: "Wooden House",
        text: [
            "You step into a wooden house.",
            "Some of the people recognize you and bring you to one of the rooms.",
            "They are friendly and talk to you."
        ],
        image: "woodenHouse.png",
        options: [
            {
                text: "Take a small nap.",
                nextId: 1,
                nextKey: "napText"
            }
        ]
    },
    {
        id: "woodenHouse",
        status: "Wooden House",
        text: [],
        image: "woodenHouse.png",
        options: []
    }
]

const tutorialTextNodes = [
    {
        id: 1,
        text: [
            "'Once upon a time...'",
            "'In a land far, far away...'",
            "'There lived a king...'"
        ],
        fontFamily: "Old English, serif",
        image: "Old English.png",
        options: [
            {
                text: "And?",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "'Ahem. Wrong book.'",
            "'Let me look up your story.'\n",
            "'I can't find the story that belongs to you.'"
        ],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "Huh?",
                nextId: 3
            }
        ],
    },
    {
        id: 3,
        text: [
            "'Hmm...'",
            "'I guess...'",
            "'You gonna make your own story.'"
        ],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "Okay",
                nextId: 4
            }
        ]
    },
    {
        id: 4,
        text: ["'Are you ready?'"],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "Yes, I'm ready.",
                nextId: 5
            },
            {
                text: "Nope, I'm not ready.",
                nextId: 6
            }
        ]
    },
    {
        id: 5,
        text: ["'Okay, let's see...'"],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "Okay",
                nextId: 7
            }
        ]
    },
    {
        id: 6,
        text: [
            "'Ha! You're not ready, are you?'",
            "'Too late.'"
        ],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "-.-",
                nextId: 7
            }
        ]
    },
    {
        id: 7,
        text: [
            "'You walked into a green field.'",
            "'There is a pink sausage at the edge of the field.'",
            "'What you gonna do?'"
        ],
        fontFamily: "Silkscreen, monospace",
        options: [
            {text: "eat", nextId: 8},
            {text: "sayang", nextId: 8},
            {text: "hug", nextId: 8},
            {text: "cuddle", nextId: 8},
            {text: "kiss", nextId: 8},
            {text: "bite", nextId: 8},
            {text: "rub", nextId: 8},
            {text: "shortcut", nextId: 8},
            {text: "sniff", nextId: 8},
            {text: "wrap", nextId: 8},
            {text: "chop", nextId: 8},
            {text: "muak", nextId: 8}
        ]
    },
    {
        id: 8,
        text: [
            "'See.'",
            "'You're ready.'\n",
            "'Let's begin!'"
        ],
        fontFamily: "Silkscreen, monospace",
        options: [
            {
                text: "Okay",
                nextId: 1,
                nextKey: "chap1"
            }
        ]
    }
]

// Chapter 1
const chap1Nodes = [
    {
        id: 1,
        text: ["Chapter 1"],
        options: [
            {
                text: "Start",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "After following your mum's tail for years...",
            "You finally decided to leave your village and go for an adventure."
        ],
        options: [
            {
                text: "Then",
                nextId: 3
            }
        ]
    },
    {
        id: 3,
        text: [
            "You walk alone across the fields and hills.",
            "You see a small farm in the distance."
        ],
        options: [
            {
                text: "Then",
                nextId: 4
            }
        ]
    },
    {
        id: 4,
        text: [
            "Suddenly, the thunder rumbles through the air.",
            "You run to the farm."
        ],
        options: [
            {
                text: "Then",
                nextId: 5
            }
        ]
    },
    {
        id: 5,
        text: [
            "The farm is way bigger than you thought.",
            "You wait for the storm to stop.\n",
            "You wait tiredly and sleep..."
        ],
        options: [
            {
                text: "Zzz",
                nextId: "barn0",
                nextKey: "location"
            }
        ]
    }
]
const pigTextNodes = [
    {
        id: 1,
        text: [
            "'Hey, you!'",
            "'I'm a pig!'"
        ],
        fontFamily: "Another Danger, sans-serif",
        options: [
            {
                text: "What?",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "You shockingly look at a pig.",
            "Pig looks more shocked than you."
        ],
        fontFamily: "Peppa Pig, sans-serif",
        options: [
            {
                text: "Why the hell are you shouting at me?!",
                nextId: 3
            },
            {
                text: "Huh? What's happening?",
                nextId: 4
            }
        ]
    },
    {
        id: 3,
        text: [
            "'I...'",
            "Pig shudders.\n",
            "'I'm sorry'"
        ],
        fontFamily: "Peppa Pig, sans-serif",
        options: [
            {
                text: "This is ridiculous. You ignore him.",
                nextId: "gardenB",
                nextKey: "location",
                setState: (player) => player.farmilyLevel = 0
            },
            {
                text: "Are you okay?",
                nextId: 4
            }
        ]
    },
    {
        id: 4,
        text: [
            "'I'm sorry.'",
            "'I am told to do this. By my friends. Truth or dare.'\n",
            "You can see how scared Pig is from his eyes."
        ],
        fontFamily: "Peppa Pig, sans-serif",
        options: [
            {
                text: "Oh okay.",
                nextId: 5
            },
            {
                text: "Alright.",
                nextId: 5
            }
        ]
    },
    {
        id: 5,
        text: [
            "Pig sees you stop talking, then runs away embarrassed.\n",
            "You still can't catch a glimpse of what happened.",
            "Head spinning..."
        ],
        fontFamily: "Peppa Pig, sans-serif",
        options: [
            {
                text: "Head spin",
                nextId: "gardenB",
                nextKey: "location",
                setState: (player) => player.farmilyLevel = 1
            }
        ]
    }
]
const sleepingPigNodes = [
    {
        id: 1,
        text: [
            "You poke into the hay.",
            "In the next second, you realise you are poking a pink sausage.",
            "It is moving!"
        ],
        options: [
            {
                text: "Huh?",
                nextId: 2,
                setState: (player) => player.wokePig = true
            }
        ]
    },
    {
        id: 2,
        text: [
            "Pig turns his head around sleepily.",
            "You are not sure what to do."
        ],
        options: [
            {
                text: "Continue poking.",
                nextId: 3,
                setState: (player) => player.farmilyLevel ++
            },
            {
                text: "Stunned and head spin.",
                nextId: 4,
                setState: (player) => player.farmilyLevel ++
            },
            {
                text: "Run before pig notices.",
                nextId: 5
            }
        ]
    },
    {
        id: 3,
        text: [
            "Pig shows his annoyed face and tries to push you away.",
            "Pig is confused when he notices it is the duck he met.\n",
            "You realise it is too weird to do this to a stranger, then you run away."
        ],
        options: [
            {
                text: "Run away",
                nextId: "gardenA",
                nextKey: "location"
            }
        ]
    },
    {
        id: 4,
        text: [
            "Pig and you are full of confusion.",
            "You say 'I'm sorry' and walk away."
        ],
        options: [
            {
                text: "Walk away",
                nextId: "gardenA",
                nextKey: "location"
            }
        ]
    },
    {
        id: 5,
        text: [
            "You run out from the barn.",
            "Not sure if pig saw you."
        ],
        options: [
            {
                text: "Keep running",
                nextId: "gardenA",
                nextKey: "location"
            }
        ]
    }
]
const lunaTextNodes = [
    {
        id: 1,
        text: [
            "You walk towards the little cats.",
            "A girl is patting a gray cat.\n",
            "As you approaching her, she notice you from the side of her eyes."
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "Pat the cat.",
                nextId: 2
            },
            {
                text: "Hi with a big smile.",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "'Hi.' says the girl",
            "'I'm Luna.'"
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "I'm new here.",
                nextId: 3,
                setState: (player) => player.metCharacterIds.push("luna")
            }
        ]
    },
    {
        id: 3,
        text: [
            "Another white cat with a black spot on her face walking slowly to your side.",
            "You pat the white cat.",
            "Luna is smiling, just so are you."
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "Enjoy the moment.",
                nextId: 4
            }
        ]
    },
    {
        id: 4,
        text: ["'You lives here?'"],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "No, I just came here to explore new places.",
                nextId: 5
            }
        ]
    },
    {
        id: 5,
        text: [
            "'Oh. I see.'",
            "'My family brought me here a few times when I was a kid.'"
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "That's nice.",
                nextId: 6
            }
        ]
    },
    {
        id: 6,
        text: [
            "'And now, I'm here because I had a fight with my parents.'\n",
            "'I don't think I can get back anytime soon.'"
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "I'm sorry.",
                nextId: 7
            },
            {
                text: "Oh... It's okay to get some air out here.",
                nextId: 7
            }
        ]
    },
    {
        id: 7,
        text: [
            "Luna doesn't seem upset as she pats the cat.\n",
            "'Do you want me to bring you around?' Luna asks.",
            "'I know somewhere beautiful to go.'"
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "Yes, sure!",
                nextId: 8
            },
            {
                text: "I will think about it.",
                nextId: 8
            }
        ]
    },
    {
        id: 8,
        text: [
            "'Meet me tonight here in the garden.'",
            "You can see how happy Luna is.\n",
            "'I gonna have something to eat near the pond now.'"
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "Okay.",
                nextId: 9
            }
        ]
    },
    {
        id: 9,
        text: [
            "Luna waves her hand.",
            "Then, hops away."
        ],
        fontFamily: "Austein, cursive",
        options: [
            {
                text: "Okay.",
                nextId: "gardenA",
                nextKey: "location"
            }
        ]
    }
]
const catTextNodes = [
    {
        id: 1,
        text: [
            "You walk towards the little cats.",
            "The white cat with a black spot seems to recognize you."
        ],
        options: [
            {
                text: "Pat the cat.",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "The white cat wraps itself around your leg.",
            "At the same time, other cats are approaching you."
        ],
        options: [
            {
                text: "Rub the cat belly.",
                nextId: 3
            },
            {
                text: "Pat every cat one by one.",
                nextId: 3
            }
        ]
    },
    {
        id: 3,
        text: [
            "After some time, you realise you lost the sense of time.",
            "You enjoy every second of it."
        ],
        options: [
            {
                text: "You're done.",
                nextId: "gardenA",
                nextKey: "location",
                setState: (player) => player.catPatting = true
            }
        ]
    }
]
const foodTextNodes = [
    {
        id: 1,
        text: [
            "For a second, you step towards the pond.",
            "Then you see a food stall by the pond."
        ],
        options: [
            {
                text: "Walk to the food stall.",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "You see a bunch of food on the shelves.",
            "What catches your attention is the name of the stall.",
            "It is 'Food for the Hungry'."
        ],
        options: [
            {
                text: "Ask for a bowl of noodle.",
                nextId: 4
            },
            {
                text: "Ask for a bowl of fish porridge.",
                nextId: 4
            },
            {
                text: "Walk away. Not hungry.",
                nextId: 3
            }
        ]
    },
    {
        id: 3,
        text: [
            "The old man behind the stall smiles.",
            "He turns to his back, then still hands you a bowl of warm noodle."
        ],
        options: [
            {
                text: "Thank you very much!",
                nextId: 5
            }
        ]
    },
    {
        id: 4,
        text: [
            "The old man behind the stall smiles.",
            "You feel your belly rumbling as the old man making your food.\n",
            "Very quickly, you are handed with a bowl of food."
        ],
        options: [
            {
                text: "Thank you!",
                nextId: 5
            }
        ]
    },
    {
        id: 5,
        text: [
            "There is no table, only tree stumps.",
            "You find a seat in the back of the stall.",
            "You sit down and enjoy the food."
        ],
        options: [
            {
                text: "Enjoy the food.",
                nextId: "foodStall",
                nextKey: "location",
                setState: (player) => player.metCharacterIds.push("oldMan")
            }
        ]
    }
]
const oldManTextNodes = [
    {
        id: 1,
        text: [
            "You greet the old man.\n",
            "'Hi. How's your day?' Old man smiles."
        ],
        options: [
            {
                text: "Great!",
                nextId: 2
            },
            {
                text: "Not bad.",
                nextId: 2
            }
        ]
    },
    {
        id: 2,
        text: [
            "You ask old man about his day.\n",
            "'I was busy an hour ago.' says old man",
            "'Not much now.'"
        ],
        options: [
            {
                text: "There are so many people.",
                nextId: 3
            },
            {
                text: "Great! I got to go.",
                nextId: 5
            }
        ]
    },
    {
        id: 3,
        text: [
            "'Yes. Groups of wanderers pass by today.'\n",
            "'Usually, there are only a few people here.'",
            "'Are you one of them?'"
        ],
        options: [
            {
                text: "Yes. But alone.",
                nextId: 4
            },
            {
                text: "No.",
                nextId: 4
            }
        ]
    },
    {
        id: 4,
        text: [
            "Old man points to the group of black clothing people.",
            "They are not good people. They are shady."
        ],
        options: [
            {
                text: "Listen and nod.",
                nextId: 5
            }
        ]
    },
    {
        id: 5,
        text: [
            "Old man gets back to his food stall and does his work.",
            "You don't disturb him and say bye to him.\n",
            "'I'm glad you're here. I'm looking forward to seeing you again.' old man says to you."
        ],
        options: [
            {
                text: "Walk away.",
                nextId: "foodStall",
                nextKey: "location"
            }
        ]
    }
]
const adventurerTextNodes = [
    {
        id: 1,
        text: [
            "'We travel all over the world.'",
            "'Now we are staying in this farm for sometimes.'",
            "'Want to join us? You can only choose one group.'"
        ],
        options: [
            {
                text: "Not yet. Let me ask around first.",
                nextId: "foodStall",
                nextKey: "location",
                requiredState: (player) => !player.metCharacterIds.includes("bandit")
            },
            {
                text: "Yes",
                nextId: 2,
                requiredState: (player) => player.metCharacterIds.includes("bandit")
            },
            {
                text: "No",
                nextId: "foodStall",
                nextKey: "location",
                requiredState: (player) => player.metCharacterIds.includes("bandit")
            }
        ]
    },
    {
        id: 2,
        text: [
            "'Great! We are going to the market tomorrow.'",
            "'We will stay in those houses for a while.'",
            "One of them points to the wooden houses across the field."
        ],
        options: [
            {
                text: "Okay",
                nextId: "foodStall",
                nextKey: "location",
                setState: (player) => player.group = "adventurer"
            }
        ]
    }
]
const banditTextNodes = [
    {
        id: 1,
        text: [
            "'We help the people in need.' says a girl with dark hood on.",
            "'We are not police but we do what they fail to do.'",
            "'Want to join us? You can only choose one group.'"
        ],
        options: [
            {
                text: "Not yet. Let me ask around first.",
                nextId: "foodStall",
                nextKey: "location",
                requiredState: (player) => !player.metCharacterIds.includes("adventurer")
            },
            {
                text: "Yes",
                nextId: 2,
                requiredState: (player) => player.metCharacterIds.includes("adventurer")
            },
            {
                text: "No",
                nextId: "foodStall",
                nextKey: "location",
                requiredState: (player) => player.metCharacterIds.includes("adventurer")
            }
        ]
    },
    {
        id: 2,
        text: [
            "'Nice to have you here.'",
            "'Tomorrow we have important work to do. You will be joining us.'",
            "'We will stay in the wooden houses across the field.'"
        ],
        options: [
            {
                text: "Okay",
                nextId: "foodStall",
                nextKey: "location",
                setState: (player) => player.group = "bandit"
            }
        ]
    }
]
const napTextNodes = [
    {
        id: 1,
        text: [
            "You take a nap and..."
        ],
        options: [
            {
                text: "...",
                nextId: 2,
                setState: (player) => player.chapter = 2
            }
        ]
    },
    {
        id: 2,
        text: ["End of Chapter 1"],
        options: []
    }
]

// Chapter 2
// TODO: Old man injured

class Curtain {
    constructor(id) {
        this.curtain = document.getElementById(id);
    }

    get backgroundImage() {
        return this.curtain.style.backgroundImage;
    }

    set backgroundImage(image) {
        this.curtain.style.backgroundImage = image;
    }

    open() {
        setTimeout(
            () => {
                this.curtain.style.transition = "opacity 5s";
                this.curtain.style.opacity = 0.2;
            },
            0
        );
    }

    close() {
        this.curtain.style.transition = "none";
        this.curtain.style.opacity = 0;
    }
}

class StatusBar {
    constructor(id) {
        this.statusBar = document.getElementById(id);
    }

    get text() {
        return this.statusBar.textContent;
    }

    set text(text) {
        this.statusBar.textContent = text;
    }

    hide() {
        this.statusBar.style.transition = "none";
        this.statusBar.style.opacity = 0;
    }

    show() {
        setTimeout(
            () => {
                this.statusBar.style.transition = "opacity 2s";
                this.statusBar.style.opacity = 1;
            },
            2000
        );
    }
}

class Display {
    constructor(id) {
        this.element = document.getElementById(id);
    }

    typePrint(text, speed = 1) {
        const recursiveType = () => {
            if (i < text.length) {
                this.print(text.charAt(i));
                i++;

                const timeout = 50 / (text.charAt(i) === "\n" ? speed / 10 : speed);
                setTimeout(recursiveType, timeout);
            }
        }

        let i = 0;
        recursiveType();
    }

    print(text) {
        this.element.textContent += text;
    }

    removeLast(count, speed = 1) {
        const recursiveRemove = () => {
            if (i < count) {
                this.removeLastChar();
                i++;

                const timeout = 50 / (speed);
                setTimeout(recursiveRemove, timeout);
            }
        }

        let i = 0;
        recursiveRemove();
    }

    removeLastChar() {
        this.element.textContent = this.element.textContent.slice(0, -1);
    }

    clear() {
        this.element.textContent = "";
    }
}

class Input {
    constructor(id) {
        this.input = document.getElementById(id);
    }

    get value() {
        return this.input.value;
    }

    clear() {
        this.input.value = "";
    }

    show() {
        this.input.classList.remove("zero-height");
    }

    focus() {
        this.input.focus();
    }

    addEventListener(type, listener, closeEventCode = null) {
        this.input.addEventListener(
            type,
            (e) => {
                if (!closeEventCode || e.code === closeEventCode) {
                    listener(e.code);
                }
            }
        );
    }
}

class ButtonGrid {
    constructor(id) {
        this.grid = document.getElementById(id);
    }

    firstButton() {
        return this.grid.querySelector(".button:not(.hidden)");
    }

    focusFirstButton() {
        const firstButton = this.firstButton();
        if (firstButton) {
            firstButton.classList.add("default");
        }
    }

    removeAllButtons() {
        this.grid.textContent = "";
    }

    hide() {
        this.grid.style.transition = "none";
        this.grid.style.opacity = 0;
    }

    show() {
        setTimeout(
            () => {
                this.grid.style.transition = "opacity 2s ease";
                this.grid.style.opacity = 1;
                this.focusFirstButton();
            },
            2000
        );
    }

    addButton(text, onClick) {
        const button = document.createElement("button");
        const empty = text === "";

        button.textContent = empty ? "<Enter>" : text;
        button.classList.add("button", "half-width");
        if (empty) {
            button.classList.add("italic");
        }

        this.grid.appendChild(button);

        setTimeout(
            () => button.addEventListener("click", onClick),
            2000
        );
    }

    filterButtons(text) {
        const buttons = this.grid.querySelectorAll(".button");

        buttons.forEach(
            (button) => {
                if (button.textContent.toLowerCase().includes(text.toLowerCase())) {
                    button.classList.remove("hidden", "default");
                } else {
                    button.classList.add("hidden");
                }
            }
        );

        this.focusFirstButton();
    }
}

class Game {
    constructor() {
        this.curtain = new Curtain("curtain");
        this.statusBar = new StatusBar("status-bar");
        this.display = new Display("display");
        this.input = new Input("input");
        this.buttonGrid = new ButtonGrid("button-grid");

        this.nodes = {
            location: locationNodes,
            tutorialText: tutorialTextNodes,
            chap1: chap1Nodes,
            pigText: pigTextNodes,
            sleepingPig: sleepingPigNodes,
            lunaText: lunaTextNodes,
            catText: catTextNodes,
            foodText: foodTextNodes,
            oldManText: oldManTextNodes,
            adventurerText: adventurerTextNodes,
            banditText: banditTextNodes,
            napText: napTextNodes
        };
        this.player = player;
        this.key = "tutorialText"
        this.chapter = player.chapter;

        this.changeFont("Silkscreen, monospace");
        document.addEventListener(
            "click",
            () => this.welcome(),
            {once: true}
        )
        document.addEventListener(
            "click",
            () => this.input.focus()
        )

        // TODO: Uncomment this
        // this.key = "chap1";
        // this.start(5);
    }

    writeJson(obj, fileName) {
        const json = JSON.stringify(obj);
        const a = document.createElement("a");
        const file = new Blob(
            [json], {type: "text/plain"}
        );
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
    }

    changeFont(font) {
        const fontSize = {
            "Silkscreen, monospace": "30px",  // Narrator
            "Old English, serif": "35px",
            "Varela Round, sans-serif": "25px",  // Duck
            "Another Danger, sans-serif": "35px",
            "Peppa Pig, sans-serif": "30px",  // Pig
            "Austein, cursive": "30px"  // Luna
        }

        document.querySelectorAll("body, input").forEach(
            (element) => {
                element.style.fontFamily = font;
                element.style.fontSize = fontSize[font];
            }
        );
    }

    welcome() {
        this.display.typePrint("Hello world!");
        setTimeout(
            () => this.display.removeLast(6), 2000
        )
        setTimeout(
            () => this.display.typePrint(" duck!"),
            3000
        )
        setTimeout(() => this.start(), 5000)
    }

    clear() {
        this.display.clear();
        this.input.clear();
        this.buttonGrid.removeAllButtons();
    }

    show(index) {
        const node = this.nodes[this.key].find(
            node => node.id === index
        );

        // Change font
        this.changeFont(
            node.fontFamily
                ? node.fontFamily
                : "Varela Round, sans-serif"
        );

        const showAll = () => {
            // Show background
            const image = node.image
                ? `url("image/${node.image}")`
                : "none";
            if (this.curtain.backgroundImage !== image) {
                this.curtain.close();
                this.curtain.backgroundImage = image;
                this.curtain.open();
            }

            // Change status
            const status = node.status ? node.status : "";
            if (this.statusBar.text !== status) {
                this.statusBar.hide();
                this.statusBar.text = status;
                this.statusBar.show();
            }

            // Show display text
            const speed = this.key.includes("Text") ? 1 : 2;
            this.display.typePrint(node.text.join("\n"), speed);

            // Show buttons
            this.buttonGrid.hide();
            this.buttonGrid.removeAllButtons();
            node.options.forEach(
                option => {
                    if (!option.requiredState || option.requiredState(this.player)) {
                        this.buttonGrid.addButton(
                            option.text,
                            () => this.clearAndShow(option.nextId, option.nextKey, option.setState)
                        );
                    }
                }
            );
            this.buttonGrid.show();
        }
        setTimeout(showAll, 0);
    }

    clearAndShow(index, key, setState) {
        this.clear();

        const oldKey = this.key;
        this.key = key || this.key;
        const visitedLocationIds = this.player.visitedLocationIds;
        if (this.key === "location" && !visitedLocationIds.includes(index)) {
            visitedLocationIds.push(index);
        }
        if (setState) {
            setState(this.player);
        }
        console.log(this.player);
        if (this.player.chapter !== this.chapter) {
            const fileName = `player_${this.player.chapter}.json`;
            this.writeJson(this.player, fileName);
        }

        setTimeout(
            () => this.show(index),
            this.key !== oldKey ? 2000 : 0
        )
    }

    keyDownListener() {
        this.buttonGrid.firstButton().click();
    }

    keyUpListener() {
        const inputValue = this.input.value;
        this.buttonGrid.filterButtons(inputValue);
    }

    start(index = 1) {
        let started = false;
        const startListener = () => {
            if (!started) {
                started = true;
                this.clearAndShow(index);
                this.input.addEventListener(
                    "keydown",
                    () => this.keyDownListener(),
                    "Enter"
                );
                this.input.addEventListener(
                    "keyup",
                    () => this.keyUpListener()
                );
            }
        };

        this.input.addEventListener(
            "keydown", startListener, "Enter"
        );

        this.input.show();
    }
}

window.onload = () => new Game();
